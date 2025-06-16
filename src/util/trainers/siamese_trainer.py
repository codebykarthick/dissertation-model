import os
from datetime import datetime
from typing import cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from util.constants import CONSTANTS
from util.data_loader import (
    SiameseDataset,
    generate_eval_transforms,
    generate_train_transforms,
)
from util.trainers.trainer import Trainer


class SiameseTrainer(Trainer):
    def __init__(self, roi: bool, fill_noise: bool, model_name: str, num_workers: int, k: int,
                 is_sampling_weighted: bool, is_loss_weighted: bool, batch_size: int, epochs: int,
                 task_type: str, lr: float, patience: int, label: str, roi_weight: str = "", delta: float = 0.02, filename=""):
        super().__init__(roi, fill_noise, model_name, num_workers, k, is_sampling_weighted,
                         is_loss_weighted, batch_size, epochs, task_type, lr, patience, label, roi_weight,
                         delta, filename)

        image_dir = os.path.join(os.getcwd(), "dataset")
        self.image_dir = image_dir

        # Base dataset for stratification
        base_dataset = SiameseDataset(
            image_dir=self.image_dir,
            roi_enabled=roi,
            roi_weight=self.roi_weight
        )
        # Prepare indices and labels for splitting images
        all_indices = list(range(len(base_dataset)))
        all_labels = [
            base_dataset.label_map[base_dataset.image_list[i]]
            for i in all_indices
        ]

        # Prepare model and create data loaders
        _, self.dimensions = self.create_model_from_name(
            self.model_name, self.task_type)

        # 80% train, 20% temp
        train_idx, temp_idx, _, temp_labels = train_test_split(
            all_indices, all_labels, test_size=0.2, stratify=all_labels, random_state=42
        )
        # Split temp into 50/50 → 10% val, 10% test
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
        )

        # Instantiate separate dataset instances for splits
        train_dataset = SiameseDataset(
            image_dir=self.image_dir,
            roi_enabled=roi,
            roi_weight=self.roi_weight,
            allowed_indices=train_idx
        )
        val_dataset = SiameseDataset(
            image_dir=self.image_dir,
            roi_enabled=roi,
            roi_weight=self.roi_weight,
            allowed_indices=val_idx
        )
        test_dataset = SiameseDataset(
            image_dir=self.image_dir,
            roi_enabled=roi,
            roi_weight=self.roi_weight,
            allowed_indices=test_idx
        )

        # Assign transforms
        train_dataset.transform = generate_train_transforms(
            self.dimensions, fill_with_noise=self.fill_noise
        )
        val_dataset.transform = generate_eval_transforms(
            self.dimensions, fill_with_noise=self.fill_noise
        )
        test_dataset.transform = generate_eval_transforms(
            self.dimensions, fill_with_noise=self.fill_noise
        )

        # Weighted sampling and triplet loss
        sampler = None
        criterion = TripletMarginLoss(margin=self.delta, p=2)

        if self.is_sampling_weighted:
            train_labels_fold = np.array(all_labels)[train_idx]
            class_counts = np.bincount(train_labels_fold)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[train_labels_fold]
            sampler = WeightedRandomSampler(
                weights=cast(list[float], sample_weights.tolist()),
                num_samples=len(sample_weights),
                replacement=True
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        self.criterion = criterion

    def train(self):
        """Standard training for Siamese few-shot classification with validation."""
        self.log.info("Starting Siamese few-shot classification training")
        model, _ = self.create_model_from_name(self.model_name, self.task_type)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        best_f1 = 0.0
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            model.train()
            total_loss = 0.0
            for anchor, positive, negative, _ in tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}', leave=False):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                optimizer.zero_grad()
                emb_a = model(anchor)
                emb_p = model(positive)
                emb_n = model(negative)
                loss = self.criterion(emb_a, emb_p, emb_n)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            # Validation
            current_f1, metrics = self.validate(model)
            scheduler.step(metrics["val_loss"])
            self.log.info(
                f"Epoch {epoch}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {metrics['val_loss']:.4f} | F1 (Current): {current_f1:.4f} | F1 (Best): {best_f1:.4f}"
            )

            if self.evaluate_and_save(current_metric=current_f1, best_metric=best_f1, model=model, metrics=metrics):
                best_f1 = current_f1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    self.log.info(f"Early stopping at epoch {epoch}")
                    break

    def validate(self, model: torch.nn.Module) -> tuple[float, dict]:
        """Run validation on triplets, compute average triplet loss and F1 on pair distances."""
        total_val_loss = 0.0
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for anchor, positive, negative, label in tqdm(self.val_loader, desc="Validation", leave=False):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                emb_a = model(anchor)
                emb_p = model(positive)
                emb_n = model(negative)
                val_loss = self.criterion(emb_a, emb_p, emb_n)
                total_val_loss += val_loss.item()

                d_pos = torch.norm(emb_a - emb_p, p=2, dim=1).cpu().tolist()
                d_neg = torch.norm(emb_a - emb_n, p=2, dim=1).cpu().tolist()

                for dp, dn, lbl in zip(d_pos, d_neg, label):
                    y_pred.append(1 if dp < dn else 0)
                    y_true.append(lbl)

        avg_val_loss = total_val_loss / len(self.val_loader)
        metrics = {"val_loss": avg_val_loss}
        f1 = cast(float, f1_score(y_true, y_pred))
        return f1, metrics

    def evaluate(self):
        """Final evaluation on the evaluation set."""
        self.log.info("Evaluating Siamese few-shot model on evaluation set")
        model, _ = self.create_model_from_name(self.model_name, self.task_type)
        model = model.to(self.device)
        self.load_model(model, self.filename)

        model.eval()
        y_true = []
        y_pred = []
        scores = []
        with torch.no_grad():
            for anchor, positive, negative, label in tqdm(self.test_loader, desc="Evaluation", leave=False):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                emb_a = model(anchor)
                emb_p = model(positive)
                emb_n = model(negative)

                d_pos = torch.norm(emb_a - emb_p, p=2, dim=1).cpu().tolist()
                d_neg = torch.norm(emb_a - emb_n, p=2, dim=1).cpu().tolist()

                for dp, dn, lbl in zip(d_pos, d_neg, label):
                    y_pred.append(1 if dp < dn else 0)
                    score = dn - dp
                    scores.append(score)
                    y_true.append(lbl)

        # compute classification metrics
        acc = accuracy_score(y_true, y_pred)
        prec_score = precision_score(y_true, y_pred)
        rec_score = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall, precision)

        metrics = {
            "accuracy": acc,
            "precision": prec_score,
            "recall": rec_score,
            "f1_score": f1,
            "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
            "pr_auc": pr_auc,
            "roc_auc": roc_auc
        }

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{self.model_name}_CLASSIFICATION_{timestamp}.json"
        self.save_results(metrics=metrics, filename=filename)
