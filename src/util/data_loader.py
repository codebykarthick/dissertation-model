import os

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
    def __init__(self, image_dir: str, roi_weight: str = "", defined_transforms=None, conf_threshold: float = 0.3):
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
        self.roi_model = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        self._create_roi_model()
        img_path = os.path.join(self.image_dir, self.images[index])
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[self.images[index]]

        if self.roi_model is not None:
            image = self._apply_roi_and_crop(image)

        if self.defined_transforms:
            image = self.defined_transforms(image)

        return image, label

    def set_transforms(self, defined_transforms):
        self.defined_transforms = defined_transforms

    def _create_roi_model(self):
        if self.roi_weight != "" and not self.roi_model:
            self.roi_model = YOLO(os.path.join(
                "weights", "yolo", self.roi_weight)).eval()

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
        positive_anchors: list[str],
        negative_anchors: list[str],
        transform=None,
        roi_weight: str = "",
        conf_threshold: float = 0.3
    ):
        super().__init__()
        self.image_dir = os.path.join(image_dir, 'Images')
        self.label_file = os.path.join(image_dir, 'labels.csv')
        self.labels_df = pd.read_csv(self.label_file)
        self.label_map = dict(
            zip(self.labels_df['file_id'], self.labels_df['is_asp_fungi']))
        self.transform = transform
        self.roi_model = None
        self.roi_weight = roi_weight

        self.image_list = sorted(os.listdir(self.image_dir))
        self.positive_anchors = [
            img for img in positive_anchors if img in self.label_map]
        self.negative_anchors = [
            img for img in negative_anchors if img in self.label_map]

        # Quick sanity assertion
        assert len(self.positive_anchors) == len(positive_anchors)
        assert len(self.negative_anchors) == len(negative_anchors)

        self.pairs = self._generate_pairs()
        self.conf_threshold = conf_threshold

    def _generate_pairs(self):
        pairs = []

        # Positive pairs: same class as anchor
        for anchor in self.positive_anchors:
            anchor_label = self.label_map[anchor]
            for candidate in self.image_list:
                if candidate != anchor and self.label_map.get(candidate) == anchor_label:
                    pairs.append((anchor, candidate, 1))

        # Negative pairs: different class
        for anchor in self.negative_anchors:
            anchor_label = self.label_map[anchor]
            for candidate in self.image_list:
                if candidate != anchor and self.label_map.get(candidate) != anchor_label:
                    pairs.append((anchor, candidate, 0))

        return pairs

    def _create_roi_model(self):
        if self.roi_weight != "" and not self.roi_model:
            self.roi_model = YOLO(os.path.join(
                "weights", "yolo", self.roi_weight)).eval()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        self._create_roi_model()
        anchor_name, candidate_name, label = self.pairs[index]
        img1 = Image.open(os.path.join(
            self.image_dir, anchor_name)).convert("RGB")
        img2 = Image.open(os.path.join(
            self.image_dir, candidate_name)).convert("RGB")

        if self.roi_model is not None:
            img1 = self._apply_roi_and_crop(img1)
            img2 = self._apply_roi_and_crop(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

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
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
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
