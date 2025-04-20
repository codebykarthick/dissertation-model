import os
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
import pandas as pd
from PIL import Image
from util.constants import CONSTANTS


class ClassificationDataset(Dataset):
    def __init__(self, image_dir: str, defined_transforms=None):
        super().__init__()
        self.image_dir = os.path.join(image_dir, 'images')
        self.label_file = os.path.join(image_dir, 'labels.csv')
        self.images = sorted(os.listdir(self.image_dir))
        self.defined_transforms = defined_transforms
        # Load the labels using pandas here from the csv
        self.labels = pd.read_csv(self.label_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = Image.open(img_path).convert("RGB")
        label = self.labels.iloc[index]['is_asp_fungi']

        if self.transforms:
            image = self.transforms(image)

        return image, label


# TODO: Implement weighted Sampling
def get_data_loaders(defined_transforms, images_path: str, dimensions: list[int, int],
                     is_sampling_weighted: bool, batch_size: int):
    dataset = ClassificationDataset(images_path, defined_transforms)
    train_size = int(CONSTANTS['train_split'] * len(dataset))
    remaining = len(dataset) - train_size
    val_size = remaining // 2
    test_size = remaining // 2
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])
    num_workers = min(4, os.cpu_count() // 2)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
