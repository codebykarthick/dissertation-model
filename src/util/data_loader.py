import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from ultralytics import YOLO

from util.constants import CONSTANTS
from util.logger import setup_logger

log = setup_logger()


class ResizeAndPad:
    """Resizes the image and fills any missing pixels with black or Gaussian noise
    """

    def __init__(self, size, fill_with_noise=False):
        self.size = size
        self.fill_with_noise = fill_with_noise

    def __call__(self, img):
        # resize while keeping aspect ratio
        img.thumbnail(self.size, Image.Resampling.LANCZOS)

        new_img = self._get_background()
        left = (self.size[0] - img.size[0]) // 2
        top = (self.size[1] - img.size[1]) // 2
        new_img.paste(img, (left, top))
        return new_img

    def _get_background(self):
        if self.fill_with_noise:
            noise = np.random.normal(loc=0.5, scale=0.2,
                                     size=(self.size[1], self.size[0], 3)) * 255
            noise = np.clip(noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noise.astype(np.uint8))
        else:
            return Image.new("RGB", self.size, (0, 0, 0))  # black


class ApplyCLAHE:
    """Applies Contrast Equalisation onto Photos for better contrast.
    """

    def __call__(self, img):
        img_np = np.array(img)
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return Image.fromarray(img_clahe)


class ClassificationDataset(Dataset):
    def __init__(self, image_dir: str, roi_enabled: bool, roi_weight: str = "", defined_transforms=None, conf_threshold: float = 0.3):
        super().__init__()
        self.image_dir = os.path.join(image_dir, 'Images')
        self.label_file = os.path.join(image_dir, 'labels.csv')
        self.images = sorted(os.listdir(self.image_dir))
        self.defined_transforms = defined_transforms
        # Load the labels using pandas here from the csv
        self.labels = pd.read_csv(self.label_file)
        self.label_map = dict(
            zip(self.labels['file_id'], self.labels['is_asp_fungi']))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.roi_weight = roi_weight
        self.conf_threshold = conf_threshold
        self.roi = roi_enabled
        self.roi_model = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        self._create_roi_model()
        img_path = os.path.join(self.image_dir, self.images[index])
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[self.images[index]]

        if self.roi_model != None:
            image = self._apply_roi_and_crop(image)

        if self.defined_transforms:
            image = self.defined_transforms(image)

        return image, label

    def set_transforms(self, defined_transforms):
        self.defined_transforms = defined_transforms

    def _create_roi_model(self):
        if self.roi and self.roi_weight != "" and self.roi_model == None:
            self.roi_model = YOLO(os.path.join(
                "weights", "yolo", self.roi_weight)).to(self.device).eval()

    def _apply_roi_and_crop(self, image: Image.Image) -> Image.Image:
        if self.roi_model is None:
            log.warning("RoI model is not initialised! Skipping crop")
            return image

        # Run YOLO on the PIL image
        results = self.roi_model(image, verbose=False)
        result = results[0]
        if not result.boxes:
            # no boxes detected → return full strip
            return image

        # pick the box with highest confidence
        boxes = result.boxes
        confidences = boxes.conf.cpu()
        best_idx = int(torch.argmax(confidences))
        best_conf = float(confidences[best_idx])

        # if top confidence is too low, skip cropping
        if best_conf < self.conf_threshold:
            return image

        # otherwise, crop the strip
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        return image.crop((x1, y1, x2, y2))


class SiameseDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        roi_enabled: bool,
        transform=None,
        roi_weight: str = "",
        conf_threshold: float = 0.3,
        allowed_indices: list[int] = []
    ):
        super().__init__()
        self.image_dir = os.path.join(image_dir, 'Images')
        self.label_file = os.path.join(image_dir, 'labels.csv')
        self.labels_df = pd.read_csv(self.label_file)
        self.label_map = dict(
            zip(self.labels_df['file_id'], self.labels_df['is_asp_fungi']))
        self.transform = transform
        self.roi = roi_enabled
        self.roi_model = None
        self.roi_weight = roi_weight
        self.image_list = sorted(os.listdir(self.image_dir))

        # If a subset of indices is provided, restrict to those images
        if len(allowed_indices) != 0:
            full_list = self.image_list
            self.image_list = [full_list[i] for i in allowed_indices]
            # restrict the label map to only these images
            self.label_map = {img: self.label_map[img]
                              for img in self.image_list}
        # Precompute positive and negative candidate lists
        self.true_candidates = [
            img for img in self.image_list if self.label_map[img] == 1]
        self.false_candidates = [
            img for img in self.image_list if self.label_map[img] == 0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_threshold = conf_threshold

    def _create_roi_model(self):
        if self.roi and self.roi_weight != "" and not self.roi_model:
            self.roi_model = YOLO(os.path.join(
                "weights", "yolo", self.roi_weight)).to(self.device).eval()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        self._create_roi_model()

        # Use the provided index for the anchor
        anchor_name = self.image_list[index]
        anchor_label = self.label_map[anchor_name]

        # Sample a positive example (same class)
        if anchor_label == 1:
            positive_name = random.choice(self.true_candidates)
            negative_name = random.choice(self.false_candidates)
        else:
            negative_name = random.choice(self.true_candidates)
            positive_name = random.choice(self.false_candidates)

        # Load images
        img_anchor = Image.open(os.path.join(
            self.image_dir, anchor_name)).convert("RGB")
        img_positive = Image.open(os.path.join(
            self.image_dir, positive_name)).convert("RGB")
        img_negative = Image.open(os.path.join(
            self.image_dir, negative_name)).convert("RGB")

        # Apply ROI cropping if enabled
        if self.roi_model is not None:
            img_anchor = self._apply_roi_and_crop(img_anchor)
            img_positive = self._apply_roi_and_crop(img_positive)
            img_negative = self._apply_roi_and_crop(img_negative)

        # Apply transforms with a shared random seed
        if self.transform:
            seed = np.random.randint(0, 2**32 - 1)
            random.seed(seed)
            torch.manual_seed(seed)
            img_anchor = self.transform(img_anchor)
            random.seed(seed)
            torch.manual_seed(seed)
            img_positive = self.transform(img_positive)
            random.seed(seed)
            torch.manual_seed(seed)
            img_negative = self.transform(img_negative)

        return img_anchor, img_positive, img_negative, anchor_label

    def _apply_roi_and_crop(self, image: Image.Image) -> Image.Image:
        if self.roi_model is None:
            log.warning("RoI model is not initialised! Skipping crop")
            return image

        # Run YOLO on the PIL image
        results = self.roi_model(image, verbose=False)
        result = results[0]
        if not result.boxes:
            # no boxes detected → return full strip
            return image

        # pick the box with highest confidence
        boxes = result.boxes
        confidences = boxes.conf.cpu()
        best_idx = int(torch.argmax(confidences))
        best_conf = float(confidences[best_idx])

        # if top confidence is too low, skip cropping
        if best_conf < self.conf_threshold:
            return image

        # otherwise, crop the strip
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        return image.crop((x1, y1, x2, y2))


def generate_train_transforms(dimensions: list[int], fill_with_noise: bool = False) -> transforms.Compose:
    """Generate the necessary transforms for Data Augmentation for training

    Args:
        dimensions (list[int]): The dimensions to resize the image to.
        fill_with_noise (bool, optional): To fill missing information with Black pixels or Gaussian noise. Defaults to False.

    Returns:
        transforms.Compose: The sequence of transforms to be applied for training data.
    """
    transform = transforms.Compose([
        ResizeAndPad(dimensions, fill_with_noise=fill_with_noise),
        ApplyCLAHE(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(dimensions, scale=(0.8, 1.0)),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    return transform


def generate_eval_transforms(dimensions: list[int], fill_with_noise: bool = False) -> transforms.Compose:
    """Generate the necessary transforms for validation and testing. Only resizes and applies no Augmentation.

    Args:
        dimensions (list[int]): The dimensions to resize the image to.
        fill_with_noise (bool, optional): To fill missing information with Black pixels or Gaussian noise. Defaults to False.

    Returns:
        transforms.Compose: The sequence of transforms to be applied for validation or testing data.
    """
    transform = transforms.Compose([
        ResizeAndPad(dimensions, fill_with_noise=fill_with_noise),
        ApplyCLAHE(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    return transform
