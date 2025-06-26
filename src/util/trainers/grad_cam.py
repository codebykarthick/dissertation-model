import os
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision import transforms
from torchvision.models import EfficientNet, ShuffleNetV2
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from models.classification.lfd_cnn import KDStudent
from util.data_loader import ClassificationDataset, generate_eval_transforms
from util.trainers.trainer import Trainer


class GradCamBench(Trainer):
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
        # Move model to device
        self.model = self.model.to(self.device)

        # Stratified 80/10/10 split of indices
        all_indices = np.arange(len(base_dataset))
        all_labels = np.array([base_dataset.label_map[img]
                              for img in base_dataset.images])

        # 80% train, 20% temp
        _, temp_idx, _, y_temp = train_test_split(
            all_indices, all_labels, test_size=0.2, stratify=all_labels, random_state=42
        )
        # Split temp into 50/50 → 10% val, 10% test
        _, test_idx, _, _ = train_test_split(
            temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        # Instantiate separate dataset instances for splits
        test_dataset = ClassificationDataset(
            self.image_dir, roi_enabled=self.roi, roi_weight=self.roi_weight)
        test_subset = torch.utils.data.Subset(test_dataset, test_idx)

        # Assign transforms
        cast(ClassificationDataset, test_subset.dataset).defined_transforms = generate_eval_transforms(
            self.dimensions, fill_with_noise=self.fill_noise
        )

        self.test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Select the threshold based on the model
        if isinstance(self.model, KDStudent):
            self.threshold = 0.5
            self.target_layer = self.model.conv3
        elif isinstance(self.model, EfficientNet):
            self.threshold = 0.56
            last_block = self.model.features[-1]
            conv_layer = last_block[0]
            self.target_layer = conv_layer
        elif isinstance(self.model, ShuffleNetV2):
            self.threshold = 0.47
            self.target_layer = self.model.conv5
        else:
            raise ValueError(
                f"Incompatible model for GradCAM encountered: {model_name}!")

    def evaluate(self):
        self.load_model(self.model, self.filename)
        self.model.eval()
        # Run Grad-CAM and save results
        save_root = os.path.join(
            os.getcwd(), "grad-cam", self.label, self.model_name)
        os.makedirs(save_root, exist_ok=True)

        # Inverse normalization to recover original image
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

        # Run Smooth Grad-CAM++ with automatic hook handling
        with torch.enable_grad():
            with SmoothGradCAMpp(
                self.model,
                target_layer=self.target_layer,
                num_samples=8,
                std=0.2
            ) as cam_extractor:
                for batch_idx, (images, labels) in enumerate(tqdm(self.test_loader, desc="Grad-CAM")):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > self.threshold).int()

                    for i in range(images.size(0)):
                        # Single image tensor [C,H,W]
                        image_tensor = images[i]

                        output = outputs[i].unsqueeze(0)
                        pred_class = int(preds[i].item())
                        label_class = int(labels[i].item())

                        # Generate CAM mask for the single-logit output (always channel 0)
                        # As it is a binary classification
                        activation_map = cam_extractor(
                            class_idx=0, scores=output)[0].cpu()
                        # Directly overlay mask without interpolation
                        unnormal_img = inv_normalize(
                            image_tensor.cpu()).clamp(0.0, 1.0)
                        result = overlay_mask(
                            to_pil_image(unnormal_img),
                            to_pil_image(activation_map[0], mode='F'),
                            alpha=0.5
                        )
                        # Mark whether the prediction was correct
                        correct = (pred_class == label_class)
                        status = "correct" if correct else "incorrect"
                        fname = f"img_{batch_idx}_{i}_pred_{pred_class}_label_{label_class}_{status}_score{probs[i].item():.2f}.png"
                        result.save(os.path.join(save_root, fname))
