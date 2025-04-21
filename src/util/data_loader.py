import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
import torch
from util.constants import CONSTANTS
from PIL import Image
import pandas as pd
import os


class ResizeAndPad:
    def __init__(self, size, fill_with_noise=False):
        self.size = size
        self.fill_with_noise = fill_with_noise

    def __call__(self, img):
        original_size = img.size  # (width, height)
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


class ClassificationDataset(Dataset):
    def __init__(self, image_dir: str, defined_transforms=None):
        super().__init__()
        self.image_dir = os.path.join(image_dir, 'images')
        self.label_file = os.path.join(image_dir, 'labels.csv')
        self.images = sorted(os.listdir(self.image_dir))
        self.defined_transforms = defined_transforms
        # Load the labels using pandas here from the csv
        self.labels = pd.read_csv(self.label_file)
        self.label_map = dict(
            zip(self.labels['file_id'], self.labels['is_asp_fungi']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[self.images[index]]

        if self.defined_transforms:
            image = self.defined_transforms(image)

        return image, label


def get_data_loaders(defined_transforms, images_path: str,
                     is_sampling_weighted: bool, batch_size: int):
    dataset = ClassificationDataset(images_path, defined_transforms)
    train_size = int(CONSTANTS['train_split'] * len(dataset))
    remaining = len(dataset) - train_size
    val_size = remaining // 2
    test_size = remaining // 2
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator)
    num_workers = min(4, os.cpu_count() // 2)
    sampler = None

    # Extract labels for the train dataset for weight calculation
    labels = [dataset.label_map[dataset.images[i]]
              for i in train_dataset.indices]
    class_sample_counts = torch.tensor(
        [(labels.count(0)), (labels.count(1))], dtype=torch.float)
    class_weights = 1. / class_sample_counts
    pos_weight = class_weights[1] / class_weights[0]

    if is_sampling_weighted:
        sample_weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, pos_weight


# TODO: CLAHE Is not used for the transform yet.


def generate_transforms(dimensions: list[int, int], fill_with_noise: bool = False):
    transform = transforms.Compose([
        ResizeAndPad(dimensions, fill_with_noise=fill_with_noise),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(dimensions, scale=(0.8, 1.0)),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    return transform
