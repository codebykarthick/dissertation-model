import argparse
import json
import os
import sys
from datetime import datetime
from typing import cast

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO

from models.lfd_cnn import LFD_CNN
from models.pretrained_models import (
    get_efficientnet_tuned,
    get_mobilenetv3_tuned,
    get_shufflenet_tuned,
)
from util.cloud_tools import auto_shutdown
from util.constants import CONSTANTS
from util.data_loader import (
    ClassificationDataset,
    ResizeAndPad,
    generate_eval_transforms,
    generate_train_transforms,
    get_data_loaders,
)
from util.logger import setup_logger

log = setup_logger()


class Runner:
    def __init__(self, model: torch.nn.Module, model_name: str, lr: float, epochs: int,
                 is_loss_weighted: bool, is_oversampled: bool,
                 batch_size: int, patience: int, dimensions: list[int], file_name: str,
                 roi: bool, roi_weight: str, fill_noise: bool, num_workers: int,
                 k: int = 10):
        self.roi = roi
        self.model = model
        self.model_name = model_name
        self.patience = patience
        self.k = k
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else "cpu"
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr

        img_path = os.path.join(os.getcwd(), 'dataset')

        self.train_loader, self.val_loader, self.test_loader, self.pos_weight = get_data_loaders(
            dimensions=dimensions, images_path=img_path,
            is_sampling_weighted=is_oversampled, batch_size=batch_size, fill_with_noise=fill_noise,
            num_workers=num_workers)

        self.dataset = ClassificationDataset(img_path)

        if roi:
            self.roi_model = YOLO(os.path.join("weights", "yolo", roi_weight))

        if is_loss_weighted:
            self.criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight)
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()

        # Using Adam optimiser
        self.optimiser = torch.optim.Adam(
            self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode='min', factor=0.5, patience=3, min_lr=10-6
        )

        self.epochs = epochs
        self.file_name = file_name
        self.fill_noise = fill_noise

    def _apply_roi_and_crop(self, images):
        pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
        results = self.roi_model(pil_images, verbose=False)
        cropped_images = []

        for img_pil, img_tensor, result in zip(pil_images, images, results):
            if result.boxes:
                box = result.boxes.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                cropped = img_pil.crop((x1, y1, x2, y2))
                resized = ResizeAndPad(
                    (img_tensor.shape[2], img_tensor.shape[1]), fill_noise)(cropped)
                cropped_images.append(transforms.ToTensor()(resized))
            else:
                cropped_images.append(img_tensor)

        return torch.stack(cropped_images).to(self.device)

    def train_with_cross_validation(self):
        """Training using k-fold cross validation to ensure equal training in all folds."""
        log.info(f"Starting {self.k}-Fold Cross Validation training")
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        dataset = self.dataset

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            log.info(f"Fold {fold + 1}/{self.k} --------------------------")

            # Subset datasets
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            # Apply appropriate transforms
            cast(ClassificationDataset, train_subset.dataset).defined_transforms = generate_train_transforms(
                self.dimensions, fill_with_noise=self.fill_noise
            )
            cast(ClassificationDataset, val_subset.dataset).defined_transforms = generate_eval_transforms(
                self.dimensions, fill_with_noise=self.fill_noise
            )

            # Dataloaders
            train_loader = DataLoader(
                train_subset, batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=True
            )

            # New model instance per fold
            model, _ = create_model_from_name(self.model_name)
            model = model.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3)
            criterion = self.criterion

            no_improve = 0
            timestamp, model_file = None, None
            best_f1 = 0.00

            for epoch in range(self.epochs):
                model.train()
                total_loss = 0.0
                for images, labels in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}', leave=False):
                    if self.roi:
                        images = self._apply_roi_and_crop(images)
                    else:
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
                        if self.roi:
                            images = self._apply_roi_and_crop(images)
                        else:
                            images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(images)
                        val_loss = criterion(
                            outputs, labels.view(-1, 1).float())
                        total_val_loss += val_loss.item()
                        preds = torch.sigmoid(outputs).cpu().numpy()
                        val_preds.extend(preds)
                        val_labels.extend(labels.cpu().numpy())

                avg_val_loss = total_val_loss / len(val_loader)
                scheduler.step(avg_val_loss)

                # Compute recall and use it for model checkpointing
                val_preds_bin = [1 if p > 0.5 else 0 for p in val_preds]
                val_labels_int = [int(l) for l in val_labels]
                current_recall = recall_score(val_labels_int, val_preds_bin)
                current_f1 = f1_score(val_labels_int, val_preds_bin)

                log.info(
                    f"[Fold {fold + 1}] Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                metrics = {
                    "accuracy": accuracy_score(val_labels_int, val_preds_bin),
                    "precision": precision_score(val_labels_int, val_preds_bin),
                    "recall": current_recall,
                    "f1_score": current_f1,
                    "val_loss": avg_val_loss
                }

                # Need to focus on all round performance for checkpointing.
                if current_f1 >= best_f1:
                    best_f1 = current_f1
                    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    model_file = f"{self.model_name}_fold{fold+1}_{timestamp}.pth"

                    self.model = model
                    self.save_model(model_file)

                    results_dir = os.path.join(os.getcwd(), "results")
                    os.makedirs(results_dir, exist_ok=True)
                    result_file = os.path.join(
                        results_dir, f"{self.model_name}_fold{fold+1}_{timestamp}_metrics.json")
                    with open(result_file, "w") as f:
                        json.dump(metrics, f, indent=4)
                    log.info(
                        f"Saved metrics for Fold {fold + 1} in: {result_file}")

                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        log.info(
                            f"Early stopping at epoch {epoch+1} for fold {fold + 1}")
                        break

    def export(self):
        """
        Load the model from weights, export the model to a mobile 
        supported architecture for local inference.
        """
        raise NotImplementedError("Export Method not implemented")

    def save_model(self, filename="sample.pth"):
        """
        Save the model weights as a pth file in weights/ directory
        """
        model_weights_dir = os.path.join(
            os.getcwd(), CONSTANTS["weights_path"], self.model_name)

        if not os.path.exists(model_weights_dir):
            log.info(f"Creating folder at: {model_weights_dir}")
            os.makedirs(model_weights_dir, exist_ok=True)

        model_filepath = os.path.join(model_weights_dir, filename)

        torch.save(self.model.state_dict(), model_filepath)
        log.info(f"Model saved in: {filename}")

    def get_model_filepath(self) -> str:
        model_weights_path = os.path.join(
            os.getcwd(), CONSTANTS['weights_path'], self.model_name, self.file_name)

        if not os.path.exists(model_weights_path):
            log.error(
                'Weights path does not exist to load model (probably no training happened).')
            sys.exit(1)

        return model_weights_path

    def load_model(self):
        """
        Load the model weights from a pth file in weights/ directory
        """
        model_filepath = self.get_model_filepath()
        log.info(f"Loading weights from: {model_filepath}")
        state_dict = torch.load(model_filepath, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        log.info(f"Model loaded successfully!")


def create_model_from_name(name: str) -> tuple[torch.nn.Module, list[int]]:
    """Create the model instance from the name of the model specified. Halts execution
    if model name is wrong.

    Args:
        name (str): Name of the model to be used.

    Returns:
        tuple[torch.nn.Module, list[int]]: Returns the instance of the model along with the dimensions to be used.
    """

    if name == "cnn":
        model = LFD_CNN()
    elif name == "mobilenetv3":
        model = get_mobilenetv3_tuned()
    elif name == "efficientnet":
        model = get_efficientnet_tuned()
    elif name == "shufflenet":
        model = get_shufflenet_tuned()
    else:
        log.error(f"{name} is not a valid model.")
        sys.exit(1)

    dimensions = [int(dim) for dim in CONSTANTS["models"][name].split("x")]

    return model, dimensions


if __name__ == "__main__":
    valid_models = ["mobilenetv3", "cnn", "efficientnet", "shufflenet"]

    parser = argparse.ArgumentParser(
        description="Run models for training or evaluation")
    parser.add_argument('--models', nargs='+', required=True,
                        help='List of models to train, save, or evaluate')
    parser.add_argument('--mode', choices=['train', 'export'],
                        required=True, help='Mode of operation: train, evaluate for performance benchmark or export for mobile app.')
    parser.add_argument('--k_fold', '-k', default=10,
                        help="Number of folds to be set for the cross validation.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for training')
    parser.add_argument('--batch', type=int, default=8,
                        help='Set the size of the batch for training and inference.')
    parser.add_argument('--fill_noise', action='store_true',
                        help="Fill missing area with Gaussian pixels or black", default=False)
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of workers for Pytorch dataloader.')
    parser.add_argument(
        '--file', type=str, help='File name of model weights to use when evaluating')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='Enable weighted loss to handle class imbalance')
    parser.add_argument('--weighted_sampling', action='store_true',
                        help='Enable weighted sampling for training data')
    parser.add_argument('--roi', action='store_true',
                        help="Use trained YOLO to detect the region before using it for classification.")
    parser.add_argument('--roi_weight', type=str,
                        help='File name of YOLO weight to use during pipeline.')
    parser.add_argument("--env", type=str, choices=["local", "cloud"], default="local",
                        help="Cloud mode has a special shutdown sequence to save resources.")
    parser.add_argument("--copy_dir", type=str,
                        help="Directory where logs and weights folder will be copied (required if env is cloud)")
    parser.add_argument("--patience", type=int,
                        help="Patience for early stopping (default = 3)", default=3)

    args = parser.parse_args()

    # Manual validation of the supplied arguments
    for model_name in args.models:
        if model_name not in valid_models:
            parser.error(
                f"Invalid model name '{model_name}'. Valid options are: {', '.join(valid_models)}")
    if args.mode == 'evaluate' or args.mode == 'export':
        if not args.file:
            parser.error(
                '--file has to be specified when the mode is evaluate or export')
    if args.weighted_loss == True:
        if args.weighted_sampling == True:
            parser.error(
                'Either one of weighted loss or weighted sampling can be set True at a time.')
    if args.roi == True:
        if not args.roi_weight:
            parser.error(
                '--roi_weight has to be provided to load the YOLO model if classfying with RoI.')
    if args.env == "cloud":
        if not args.copy_dir:
            parser.error("--copy_dir is required when env is cloud")
        elif not os.path.exists(args.copy_dir):
            log.error("The copy directory does not exist, halting execution.")
            sys.exit(1)

    list_of_models = args.models
    mode = args.mode
    lr = args.lr
    epochs = args.epochs
    weighted_loss = args.weighted_loss
    weighted_sampling = args.weighted_sampling
    file_name = args.file if args.file else ""
    batch_size = args.batch
    patience = args.patience
    roi = args.roi
    roi_weight = args.roi_weight
    fill_noise = args.fill_noise
    num_workers = args.workers
    k = args.k_fold

    for model_name in list_of_models:
        model, dimensions = create_model_from_name(model_name)
        runner = Runner(model=model, lr=lr, epochs=epochs, is_loss_weighted=weighted_loss,
                        is_oversampled=weighted_sampling, batch_size=batch_size, patience=patience,
                        dimensions=dimensions, model_name=model_name, file_name=file_name,
                        roi=roi, roi_weight=roi_weight, fill_noise=fill_noise, num_workers=num_workers, k=k)

        if mode == "train":
            if not os.path.exists("dataset/Images"):
                log.error(
                    "Dataset does not exist for training, please download using data_setup.py before training.")
                sys.exit(1)
            runner.train_with_cross_validation()
        elif mode == "export":
            runner.export()

    if args.env == "cloud":
        auto_shutdown(args.copy_dir)
