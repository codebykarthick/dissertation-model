import os
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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

        # Ensure model parameters require grad for CAM
        for param in self.model.parameters():
            param.requires_grad = True

        # Result directory
        save_root = os.path.join(
            os.getcwd(), "grad-cam", self.label, self.model_name)
        os.makedirs(save_root, exist_ok=True)

        # Inverse normalization to recover original image
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

        with SmoothGradCAMpp(
                self.model, self.target_layer,
                num_samples=8, std=0.2) as cam_extractor:
            for images, labels in tqdm(self.test_loader, desc="GradCAM Batch"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                # For binary single-logit models, always target the single output at index 0
                activation_maps = cam_extractor(0, outputs)
                # cam_extractor returns a list (one entry per target layer); extract our single layer's tensor
                # Tensor shape: [batch_size, H, W]
                activation_maps = activation_maps[0]

                for i in tqdm(range(images.size(0)), desc="Sub batch progress"):
                    img = images[i]
                    # Inverse normalize and clamp to [0,1]
                    img_reverted = inv_normalize(img).clamp(0.0, 1.0)

                    # Extract the CAM for this sample
                    cam_map = activation_maps[i]  # shape: [H, W]
                    cam_map = cam_map.cpu().detach()

                    original_img = to_pil_image(img_reverted)
                    cam_overlay = overlay_mask(
                        original_img,
                        to_pil_image(cam_map, mode='F'),
                        alpha=0.5
                    )
                    # Ensure same size
                    cam_overlay = cam_overlay.resize(original_img.size)

                    # Concatenate side-by-side
                    combined = Image.new(
                        'RGB', (original_img.width * 2, original_img.height))
                    combined.paste(original_img, (0, 0))
                    combined.paste(cam_overlay, (original_img.width, 0))

                    # Compute prediction, label, and save overlay with status and score
                    output_i = outputs[i]
                    score = torch.sigmoid(output_i).item()
                    pred = int(score > self.threshold)
                    label_i = int(labels[i].item())
                    status = "correct" if pred == label_i else "incorrect"
                    fname = f"img_{i}_pred_{pred}_{status}_score{score:.2f}.png"
                    combined.save(os.path.join(save_root, fname))
