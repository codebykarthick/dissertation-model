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
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from models.classification.pretrained_models import (
    get_efficientnet_tuned,
    get_shufflenet_tuned,
)
from util.data_loader import (
    ClassificationDataset,
    generate_eval_transforms,
    generate_train_transforms,
)
from util.trainers.trainer import Trainer


class DistillationTrainer(Trainer):
    """Inherited class for training normal classification.
    """

    def __init__(self, roi: bool, fill_noise: bool, model_name: str, num_workers: int, k: int,
                 is_sampling_weighted: bool, is_loss_weighted: bool, batch_size: int, epochs: int,
                 task_type: str, lr: float, patience: int, label: str, roi_weight: str = "", delta=0.02, filename="",
                 temperature=2.0, teacher1="", teacher2=""):
        super().__init__(roi=roi, fill_noise=fill_noise, model_name=model_name,
                         num_workers=num_workers, roi_weight=roi_weight, k=k,
                         is_sampling_weighted=is_sampling_weighted, is_loss_weighted=is_loss_weighted,
                         epochs=epochs, task_type=task_type, lr=lr, patience=patience, batch_size=batch_size,
                         label=label, delta=delta, filename=filename)

        image_dir = os.path.join(os.getcwd(), "dataset")
        self.image_dir = image_dir

        # Load the model for teacher1
        self.teacher1 = get_efficientnet_tuned().to(self.device)
        self.load_model(self.teacher1, teacher1)

        self.teacher2 = get_shufflenet_tuned().to(self.device)
        self.load_model(self.teacher2, teacher2)
        self.T = temperature

        # Freeze teachers and disable training-time behaviors (dropout/BN updates)
        for p in self.teacher1.parameters():
            p.requires_grad = False
        self.teacher1.eval()
        for p in self.teacher2.parameters():
            p.requires_grad = False
        self.teacher2.eval()

        # Normal Dataset loading
        base_dataset = ClassificationDataset(
            self.image_dir, roi_enabled=self.roi, roi_weight=self.roi_weight)
        self.model, self.dimensions = self.create_model_from_name(
            self.model_name, self.task_type)

        # Stratified 80/10/10 split of indices
        all_indices = np.arange(len(base_dataset))
        all_labels = np.array([base_dataset.label_map[img]
                              for img in base_dataset.images])

        # 80% train, 20% temp
        train_idx, temp_idx, _, y_temp = train_test_split(
            all_indices, all_labels, test_size=0.2, stratify=all_labels, random_state=42
        )
        # Split temp into 50/50 → 10% val, 10% test
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        # Instantiate separate dataset instances for splits
        train_dataset = ClassificationDataset(
            self.image_dir, roi_enabled=self.roi, roi_weight=self.roi_weight)
        val_dataset = ClassificationDataset(
            self.image_dir, roi_enabled=self.roi, roi_weight=self.roi_weight)
        test_dataset = ClassificationDataset(
            self.image_dir, roi_enabled=self.roi, roi_weight=self.roi_weight)
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(val_dataset, val_idx)
        test_subset = torch.utils.data.Subset(test_dataset, test_idx)

        # Assign transforms
        cast(ClassificationDataset, train_subset.dataset).defined_transforms = generate_train_transforms(
            self.dimensions, fill_with_noise=self.fill_noise
        )
        cast(ClassificationDataset, val_subset.dataset).defined_transforms = generate_eval_transforms(
            self.dimensions, fill_with_noise=self.fill_noise
        )
        cast(ClassificationDataset, test_subset.dataset).defined_transforms = generate_eval_transforms(
            self.dimensions, fill_with_noise=self.fill_noise
        )

        # Weighted sampling and loss
        sampler = None
        criterion = torch.nn.BCEWithLogitsLoss()

        if self.is_sampling_weighted or self.is_loss_weighted:
            # Compute class weights from base labels for train split
            train_labels_fold = all_labels[train_idx]
            class_counts = np.bincount(train_labels_fold)
            class_weights = 1.0 / class_counts

            if self.is_sampling_weighted:
                sample_weights = class_weights[train_labels_fold]
                sampler = WeightedRandomSampler(
                    weights=cast(list[float], sample_weights.tolist()),
                    num_samples=len(sample_weights),
                    replacement=True
                )
            if self.is_loss_weighted:
                pos_weight = torch.tensor(
                    class_weights[1] / class_weights[0], device=self.device
                )
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
            num_workers=0,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        self.criterion = criterion

    def train(self):
        self.log.info("Starting knowledge distillation training")
        # Model, optimizer, scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        best_f1 = 0.0
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for images, labels in tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}', leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Distill knowledge from teacher1 and teacher2
                with torch.no_grad():
                    t1_logits = self.teacher1(images)
                    t2_logits = self.teacher2(images)
                    teacher_logits = (t1_logits + t2_logits) / 2

                student_logits = self.model(images)
                # Compute teacher softened probabilities and distillation loss using BCE with logits
                teacher_prob = torch.sigmoid(teacher_logits / self.T)
                kd_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    student_logits / self.T, teacher_prob
                )
                kd_loss = (self.T * self.T) * kd_loss

                # Supervised loss
                sup_loss = self.criterion(
                    student_logits, labels.view(-1, 1).float())

                # Combined loss (equal weighting)
                loss = 0.5 * kd_loss + 0.5 * sup_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            current_f1, metrics = self.validate(self.model)
            scheduler.step(metrics["val_loss"])

            self.log.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"Val Loss: {metrics['val_loss']:.4f} | F1 (Current): {current_f1:.4f} | F1 (Best): {best_f1:.4f}"
            )

            # Save on improvement
            if self.evaluate_and_save(
                current_metric=current_f1, best_metric=best_f1, model=self.model, metrics=metrics
            ):
                best_f1 = current_f1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    self.log.info(f"Early stopping at epoch {epoch}")
                    break

    def validate(self, model) -> tuple[float, dict]:
        """Run validation and return F1 and metric dict."""
        model.eval()

        val_preds, val_labels = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = self.criterion(outputs, labels.view(-1, 1).float())
                total_val_loss += loss.item()

                preds = torch.sigmoid(outputs).view(-1).cpu().numpy()
                val_preds.extend(preds.tolist())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(self.val_loader)
        val_labels_int = [int(l) for l in val_labels]

        best_f1 = 0.0
        best_thresh = 0.5
        for t in np.linspace(0, 1, 101):
            preds_bin = [1 if p >= t else 0 for p in val_preds]
            f1 = f1_score(val_labels_int, preds_bin)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        val_preds_bin = [1 if p >= best_thresh else 0 for p in val_preds]
        current_f1 = float(f1_score(val_labels_int, val_preds_bin))

        metrics = {
            "model": self.model_name,
            "accuracy": accuracy_score(val_labels_int, val_preds_bin),
            "precision": precision_score(val_labels_int, val_preds_bin),
            "recall": recall_score(val_labels_int, val_preds_bin),
            "f1_score": current_f1,
            "val_loss": avg_val_loss,
            "threshold": best_thresh
        }
        return current_f1, metrics

    def evaluate(self):
        """Final evaluation on the test set."""
        self.log.info("Evaluating on test set")
        model, _ = self.create_model_from_name(self.model_name, self.task_type)
        model = model.to(self.device)
        self.load_model(model, self.filename)

        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                preds = torch.sigmoid(outputs).view(-1).cpu().numpy()
                test_preds.extend(preds.tolist())
                test_labels.extend(labels.cpu().numpy())

        test_labels_int = [int(l) for l in test_labels]
        precision, recall, _ = precision_recall_curve(
            test_labels_int, test_preds)
        auc_score = auc(recall, precision)

        fpr, tpr, _ = roc_curve(test_labels_int, test_preds)
        roc_auc = roc_auc_score(test_labels_int, test_preds)

        metrics = {
            "auc_score": auc_score,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_auc": roc_auc,
            "probs": test_preds,
            "labels": test_labels_int
        }

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{self.model_name}_AUC_ROC_{timestamp}.json"
        self.save_results(metrics=metrics, filename=filename)
