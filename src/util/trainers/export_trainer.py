import os
from datetime import datetime
from typing import cast

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from ultralytics import YOLO

from models.export.mobile_model import MobileInferenceModel
from util.data_loader import (
    ClassificationDataset,
    generate_eval_transforms,
    generate_train_transforms,
)
from util.logger import setup_logger
from util.trainers.trainer import Trainer

log = setup_logger()


class ExportTrainer(Trainer):
    def __init__(self, roi: bool, fill_noise: bool, model_name: str, num_workers: int, k: int,
                 is_sampling_weighted: bool, is_loss_weighted: bool, batch_size: int, epochs: int,
                 task_type: str, lr: float, patience: int, label: str, roi_weight: str = "", delta=0.02, filename=""):
        super().__init__(roi=roi, fill_noise=fill_noise, model_name=model_name,
                         num_workers=num_workers, roi_weight=roi_weight, k=k,
                         is_sampling_weighted=is_sampling_weighted, is_loss_weighted=is_loss_weighted,
                         epochs=epochs, task_type=task_type, lr=lr, patience=patience, batch_size=batch_size,
                         label=label, delta=delta, filename=filename)

        self.model_name = model_name
        self.model, self.dimensions = self.create_model_from_name(
            model_name, task_type)
        self.yolo_model = YOLO(os.path.join(
            "weights", "yolo", "yolov11s.pt")).to("cpu").eval()
        self.export_path = os.path.join("weights", "mobile")

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

    def export(self):
        yolo_script_file = os.path.join(
            "weights", "yolo", "yolov11s.torchscript")
        os.makedirs(os.path.dirname(self.export_path), exist_ok=True)

        self.yolo_model.export(
            format="torchscript",
            imgsz=self.dimensions or 224,
            optimize=True,
            nms=True,
            batch=1,
            device="cpu",
            mode="export"
        )

        log.info(f"YOLOv11 exported to {yolo_script_file}")

        yolo_torchscript_model = torch.jit.load(yolo_script_file)

        self.load_model(self.model, self.filename, "cpu")

        log.info("Creating Mobile format instance")
        mobile_model = MobileInferenceModel(
            yolo_model=yolo_torchscript_model, classifier_model=self.model)

        scripted_model = torch.jit.script(mobile_model)
        mobile_model_path = os.path.join(
            self.export_path, f"{self.model_name}_mobile.pt")
        scripted_model.save(mobile_model_path)
        log.info(f"Mobile model exported to {mobile_model_path}")
