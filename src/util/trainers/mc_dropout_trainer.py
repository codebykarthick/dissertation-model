import os
import uuid
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from util.data_loader import (
    ClassificationDataset,
    generate_eval_transforms,
    generate_train_transforms,
)
from util.trainers.trainer import Trainer


class MCDropoutTrainer(Trainer):
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

    def mc_dropout(self):
        # Enable dropout layers during evaluation
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    def evaluate(self, num_forward_passes=10):
        """Final evaluation on the test set."""
        self.log.info("Evaluating on test set with MC Dropout")
        self.load_model(self.model, self.filename)

        # Set the MC Dropout mode
        self.mc_dropout()

        results = []
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                # Perform multiple stochastic forward passes
                probs = []
                for _ in range(num_forward_passes):
                    outputs = self.model(images)
                    preds = torch.sigmoid(outputs).view(-1)
                    probs.append(preds.unsqueeze(0))  # shape: (1, batch_size)
                    # Concatenate and compute mean and std deviation
                # shape: (num_passes, batch_size)
                probs = torch.cat(probs, dim=0)
                mean_probs = probs.mean(dim=0)
                std_probs = probs.std(dim=0)
                for i in range(images.size(0)):
                    image_name = uuid.uuid4().hex + ".jpg"
                    results.append({
                        "true_label": int(labels[i].item()),
                        "probability": float(mean_probs[i].item()),
                        "uncertainty": float(std_probs[i].item()),
                        "image_name": image_name
                    })
                    self.save_image(images[i].cpu(),
                                    f"{image_name}")

        filename = f"{self.model_name}_MC_Dropout_Evaluation_{timestamp}.json"
        self.save_results(metrics=results, filename=filename)
