import os
from datetime import datetime
from typing import cast

import numpy as np
import torch
import torch.multiprocessing as mp
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from util.data_loader import (
    ClassificationDataset,
    generate_eval_transforms,
    generate_train_transforms,
)
from util.trainers.trainer import Trainer


class ClassificationCrossValidationTrainer(Trainer):
    """Inherited class for training normal classification.
    """

    def __init__(self, k: int, fill_noise: bool, is_sampling_weighted: bool, is_loss_weighted: bool, batch_size: int, num_workers: int,
                 model_name: str, epochs: int, task_type: str, lr: float, roi: bool, patience: int,
                 label: str = "", roi_weight: str = "", delta=0.02):
        super().__init__(roi=roi, fill_noise=fill_noise, model_name=model_name,
                         num_workers=num_workers, roi_weight=roi_weight, k=k,
                         is_sampling_weighted=is_sampling_weighted, is_loss_weighted=is_loss_weighted,
                         epochs=epochs, task_type=task_type, lr=lr, patience=patience, batch_size=batch_size,
                         label=label, delta=delta)

        image_dir = os.path.join(os.getcwd(), "dataset")
        self.image_dir = image_dir

    def train(self):
        """Training using k-fold cross validation to ensure equal training in all folds."""
        self.log.info(f"Starting {self.k}-Fold Cross Validation training")
        kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)
        # Load base dataset for stratification
        base_dataset = ClassificationDataset(
            self.image_dir, roi_enabled=self.roi, roi_weight=self.roi_weight)
        labels_array = np.array([base_dataset.label_map[img]
                                for img in base_dataset.images])

        for fold, (train_idx, val_idx) in enumerate(
            kf.split(X=np.zeros(len(base_dataset)), y=labels_array)
        ):
            self.log.info(
                f"Fold {fold + 1}/{self.k} --------------------------")

            # Instantiate separate dataset instances for this fold
            train_dataset = ClassificationDataset(
                self.image_dir, roi_enabled=self.roi, roi_weight=self.roi_weight)
            val_dataset = ClassificationDataset(
                self.image_dir, roi_enabled=self.roi, roi_weight=self.roi_weight)
            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            val_subset = torch.utils.data.Subset(val_dataset, val_idx)

            # New model instance per fold
            model, dimensions = self.create_model_from_name(
                self.model_name, self.task_type)
            model = model.to(self.device)

            # Apply appropriate transforms
            cast(ClassificationDataset, train_subset.dataset).defined_transforms = generate_train_transforms(
                dimensions, fill_with_noise=self.fill_noise
            )
            cast(ClassificationDataset, val_subset.dataset).defined_transforms = generate_eval_transforms(
                dimensions, fill_with_noise=self.fill_noise
            )

            sampler = None
            criterion = torch.nn.BCEWithLogitsLoss()

            if self.is_sampling_weighted or self.is_loss_weighted:
                # Compute class weights from labels_array for this fold
                train_labels_fold = labels_array[train_idx]
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
                    criterion = torch.nn.BCEWithLogitsLoss(
                        pos_weight=pos_weight)

            # Dataloaders
            train_loader = DataLoader(
                train_subset,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=self.num_workers,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=self.batch_size, shuffle=False,
                num_workers=0, pin_memory=True
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3)

            no_improve = 0
            best_f1 = 0.00

            for epoch in range(self.epochs):
                model.train()
                total_loss = 0.0
                for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}', leave=False):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels.view(-1, 1).float())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)

                # Validation
                model.eval()
                val_preds = []
                val_labels = []
                total_val_loss = 0.0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(images)
                        val_loss = criterion(
                            outputs, labels.view(-1, 1).float())
                        total_val_loss += val_loss.item()
                        preds = torch.sigmoid(outputs).view(-1).cpu().numpy()
                        val_preds.extend(preds.tolist())
                        val_labels.extend(labels.cpu().numpy())

                avg_val_loss = total_val_loss / len(val_loader)
                scheduler.step(avg_val_loss)

                # Compute f1 and use it for model checkpointing
                val_preds_bin = [1 if p > 0.5 else 0 for p in val_preds]
                val_labels_int = [int(l) for l in val_labels]
                current_f1 = cast(float, f1_score(
                    val_labels_int, val_preds_bin))

                self.log.info(
                    f"[Fold {fold + 1}] Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | F1 (Current): {current_f1:.4f} | F1 (Best): {best_f1:.4f}")

                metrics = {
                    "model": self.model_name,
                    "fold": (fold + 1),
                    "accuracy": accuracy_score(val_labels_int, val_preds_bin),
                    "precision": precision_score(val_labels_int, val_preds_bin),
                    "recall": recall_score(val_labels_int, val_preds_bin),
                    "f1_score": current_f1,
                    "val_loss": avg_val_loss
                }

                if self.evaluate_and_save(current_metric=current_f1, best_metric=best_f1, model=model,
                                          metrics=metrics, fold=fold, save_model=False, retain_last=1):
                    # Update our best and reset counter.
                    best_f1 = current_f1
                    no_improve = 0
                else:
                    # Update counter.
                    no_improve += 1
                    if no_improve >= self.patience:
                        self.log.info(
                            f"Early stopping at epoch {epoch+1} for fold {fold + 1}")
                        break


class ClassificationTrainer(Trainer):
    def __init__(self, roi: bool, fill_noise: bool, model_name: str, num_workers: int, k: int,
                 is_sampling_weighted: bool, is_loss_weighted: bool, batch_size: int, epochs: int,
                 task_type: str, lr: float, patience: int, label: str, roi_weight: str = "", delta=0.02, filename=""):
        super().__init__(roi=roi, fill_noise=fill_noise, model_name=model_name,
                         num_workers=num_workers, roi_weight=roi_weight, k=k,
                         is_sampling_weighted=is_sampling_weighted, is_loss_weighted=is_loss_weighted,
                         epochs=epochs, task_type=task_type, lr=lr, patience=patience, batch_size=batch_size,
                         label=label, delta=delta, filename=filename)

        image_dir = os.path.join(os.getcwd(), "dataset")
        self.image_dir = image_dir
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
        """Standard training on train set with validation at each epoch."""
        self.log.info("Starting classification training")
        # Model, optimizer, scheduler
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
            for images, labels in tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}', leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels.view(-1, 1).float())
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

            # Save on improvement
            if self.evaluate_and_save(
                current_metric=current_f1, best_metric=best_f1, model=model, metrics=metrics
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
        val_preds_bin = [1 if p > 0.5 else 0 for p in val_preds]
        val_labels_int = [int(l) for l in val_labels]
        current_f1 = float(f1_score(val_labels_int, val_preds_bin))

        metrics = {
            "model": self.model_name,
            "accuracy": accuracy_score(val_labels_int, val_preds_bin),
            "precision": precision_score(val_labels_int, val_preds_bin),
            "recall": recall_score(val_labels_int, val_preds_bin),
            "f1_score": current_f1,
            "val_loss": avg_val_loss
        }
        return current_f1, metrics

    def evaluate(self):
        """Final evaluation on the test set."""
        self.log.info("Evaluating on test set")
        model, _ = self.create_model_from_name(self.model_name, self.task_type)
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
