import os
from typing import cast

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from util.constants import CONSTANTS
from util.data_loader import (SiameseDataset, generate_eval_transforms,
                              generate_train_transforms)
from util.trainers.trainer import Trainer


class SiameseTrainer(Trainer):
    def __init__(self, roi: bool, fill_noise: bool, model_name: str, num_workers: int, k: int,
                 is_sampling_weighted: bool, is_loss_weighted: bool, batch_size: int, epochs: int,
                 task_type: str, lr: float, patience: int, roi_weight: str = ""):
        super().__init__(roi, fill_noise, model_name, num_workers, k, is_sampling_weighted,
                         is_loss_weighted, batch_size, epochs, task_type, lr, patience, roi_weight)

        image_dir = os.path.join(os.getcwd(), "dataset")
        positive_anchors = CONSTANTS["siamese"]["positive_anchors"]
        negative_anchors = CONSTANTS["siamese"]["negative_anchors"]
        dataset = SiameseDataset(
            image_dir=image_dir, positive_anchors=positive_anchors,
            negative_anchors=negative_anchors)
        # Prepare indices and labels for splitting siamese pairs
        all_indices = list(range(len(dataset)))
        all_labels = [label for _, _, label in dataset.pairs]

        # Prepare model and create data loaders
        self.model, self.dimensions = self.create_model_from_name(
            self.model_name, self.task_type)

        # 80% train, 20% temp
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(
            all_indices, all_labels, test_size=0.2, stratify=all_labels, random_state=42
        )
        # Split temp into 50/50 → 10% val, 10% test
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
        )

        # Create subsets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        test_subset = torch.utils.data.Subset(dataset, test_idx)

        # Assign transforms
        cast(SiameseDataset, train_subset.dataset).transform = generate_train_transforms(
            self.dimensions, fill_with_noise=self.fill_noise
        )
        cast(SiameseDataset, val_subset.dataset).transform = generate_eval_transforms(
            self.dimensions, fill_with_noise=self.fill_noise
        )
        cast(SiameseDataset, test_subset.dataset).transform = generate_eval_transforms(
            self.dimensions, fill_with_noise=self.fill_noise
        )

        # Weighted sampling and loss
        sampler = None
        criterion = torch.nn.BCEWithLogitsLoss()

        if self.is_sampling_weighted or self.is_loss_weighted:
            train_labels = np.array([dataset.pairs[i][2] for i in train_idx])
            class_counts = np.bincount(train_labels)
            class_weights = 1. / class_counts
            sample_weights = class_weights[train_labels]
            if self.is_sampling_weighted:
                sampler = WeightedRandomSampler(
                    weights=cast(list[float], sample_weights.tolist()),
                    num_samples=len(sample_weights),
                    replacement=True
                )
            else:
                pos_weight = torch.tensor(
                    class_weights[1] / class_weights[0], device=self.device)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # DataLoaders
        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.criterion = criterion

    def train(self):
        """Standard training for Siamese few-shot classification with validation."""
        self.log.info("Starting Siamese few-shot classification training")
        model, _ = self.create_model_from_name(self.model_name, self.task_type)
        model = model.to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        best_f1 = 0.0
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            model.train()
            total_loss = 0.0
            for img1, img2, label in tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}', leave=False):
                if self.roi:
                    img1 = self._apply_roi_and_crop(img1)
                    img2 = self._apply_roi_and_crop(img2)
                else:
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                label = label.to(self.device)

                optimizer.zero_grad()
                outputs = model(img1, img2)
                loss = criterion(outputs.view(-1), label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            # Validation
            current_f1, metrics = self.validate()
            scheduler.step(metrics["val_loss"])
            self.log.info(
                f"Epoch {epoch}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {metrics['val_loss']:.4f} | F1: {metrics['f1_score']:.4f}"
            )

            if self.evaluate_and_save(current_metric=current_f1, best_metric=best_f1, model=model, metrics=metrics):
                best_f1 = current_f1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    self.log.info(f"Early stopping at epoch {epoch}")
                    break

    def validate(self) -> tuple[float, dict]:
        """Run validation on Siamese pairs and return F1 and metric dict."""
        model = getattr(self, "model", None)
        if model is None:
            model, _ = self.create_model_from_name(
                self.model_name, self.task_type)
            model = model.to(self.device)
        model.eval()
        all_preds, all_labels = [], []
        total_val_loss = 0.0
        criterion = torch.nn.BCEWithLogitsLoss()
        with torch.no_grad():
            for img1, img2, label in tqdm(self.val_loader, desc="Validation", leave=False):
                if self.roi:
                    img1 = self._apply_roi_and_crop(img1)
                    img2 = self._apply_roi_and_crop(img2)
                else:
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                label = label.to(self.device)

                outputs = model(img1, img2)
                val_loss = criterion(outputs.view(-1), label)
                total_val_loss += val_loss.item()

                probs = torch.sigmoid(outputs.view(-1)).cpu().numpy()
                preds = [1 if p > 0.5 else 0 for p in probs]
                all_preds.extend(preds)
                all_labels.extend(label.cpu().numpy().tolist())

        avg_val_loss = total_val_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1 = cast(float, f1_score(all_labels, all_preds))
        metrics = {
            "val_loss": avg_val_loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
        return f1, metrics

    def evaluate(self):
        """Final evaluation on the evaluation set."""
        self.log.info("Evaluating Siamese few-shot model on evaluation set")
        model = getattr(self, "model", None)
        if model is None:
            self.log.warning("Model not found for evaluation")
            return
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for img1, img2, label in tqdm(self.test_loader, desc="Evaluation", leave=False):
                if self.roi:
                    img1 = self._apply_roi_and_crop(img1)
                    img2 = self._apply_roi_and_crop(img2)
                else:
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)

                outputs = model(img1, img2)
                probs = torch.sigmoid(outputs.view(-1)).cpu().numpy()
                preds = [1 if p > 0.5 else 0 for p in probs]
                all_preds.extend(preds)
                all_labels.extend(label.numpy().tolist())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        self.log.info(
            f"Evaluation Results | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"
        )
