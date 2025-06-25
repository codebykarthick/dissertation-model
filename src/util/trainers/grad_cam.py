import os
from typing import cast

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

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
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        # Depending on the model name choose the target layer.
        model_name_lower = self.model_name.lower()
        if "kdstudent" in model_name_lower:
            self.target_layer = cast(torch.nn.Module, self.model.conv3)
            self.threshold = 0.5
        elif "efficientnet" in model_name_lower:
            self.target_layer = cast(torch.nn.Module, self.model.features[-1])
            self.threshold = 0.56
        elif "shufflenet" in model_name_lower:
            self.target_layer = cast(torch.nn.Module, self.model.conv5)
            self.threshold = 0.47
        else:
            # fallback to last convolutional layer
            conv_layers = [m for m in self.model.modules(
            ) if isinstance(m, torch.nn.Conv2d)]
            self.target_layer = cast(torch.nn.Module, conv_layers[-1])
            self.threshold = 0.5

    def train(self):
        raise NotImplementedError("Wrong method invoked!")

    def export(self):
        raise NotImplementedError("Wrong method invoked!")

    def evaluate(self):
        # Move model to device, load weights, and set to evaluation mode
        self.model = self.model.to(self.device)
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

        # Run Grad-CAM with automatic hook handling
        with GradCAM(self.model, target_layer=self.target_layer) as cam_extractor:
            for batch_idx, (images, _) in enumerate(tqdm(self.test_loader, desc="Grad-CAM")):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > self.threshold).int()

                for i in range(images.size(0)):
                    img_tensor = images[i].unsqueeze(0)
                    output = outputs[i].unsqueeze(0)
                    pred_class = int(preds[i].item())

                    # Generate CAM mask
                    activation_map = cam_extractor(
                        class_idx=pred_class, scores=output
                    )[0].cpu()
                    heatmap = to_pil_image(activation_map)

                    # Recover original image from normalized tensor
                    unnorm_img = inv_normalize(images[i].cpu())
                    orig_img = to_pil_image(unnorm_img)

                    # Overlay heatmap
                    overlay = overlay_mask(orig_img, heatmap, alpha=0.5)

                    # Save overlay
                    fname = f"img_{batch_idx}_{i}_pred{pred_class}_score{probs[i].item():.2f}.png"
                    overlay.save(os.path.join(save_root, fname))
