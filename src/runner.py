import argparse
import json
import os
import sys
from datetime import datetime

import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO

from models.lfd_cnn import LFD_CNN
from models.pretrained_models import (get_efficientnet_tuned,
                                      get_mobilenetv3_tuned)
from util.cloud_tools import auto_shutdown
from util.constants import CONSTANTS
from util.data_loader import ResizeAndPad, get_data_loaders
from util.logger import setup_logger

log = setup_logger()


class Runner:
    def __init__(self, model: torch.nn.Module, model_name: str, lr: float, epochs: int,
                 is_loss_weighted: bool, is_oversampled: bool,
                 batch_size: int, patience: int, dimensions: list[int], file_name: str,
                 min_loss: float, roi: bool, roi_weight: str, fill_noise: bool):
        self.roi = roi
        self.min_loss = min_loss
        self.model = model
        self.model_name = model_name
        self.patience = patience
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else "cpu"
        img_path = os.path.join(os.getcwd(), 'dataset')

        self.train_loader, self.val_loader, self.test_loader, self.pos_weight = get_data_loaders(
            dimensions=dimensions, images_path=img_path,
            is_sampling_weighted=is_oversampled, batch_size=batch_size, fill_with_noise=fill_noise)

        if roi:
            self.roi_model = YOLO(os.path.join("weights", "yolo", roi_weight))

        if is_loss_weighted:
            self.criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight)
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)
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

    def train(self):
        """
        Run the training loop for the loaded model for the specified epochs
        """
        log.info(
            f"Training {self.model_name} model, for {self.epochs} epochs.")
        self.model.to(self.device)
        self.model.train()

        best_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for images, labels in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}', leave=False):
                if self.roi:
                    images = self._apply_roi_and_crop(images)
                else:
                    images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                # Ensure labels are float for BCELoss
                loss = self.criterion(outputs, labels.view(-1, 1).float())
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)

            # Validate after each epoch
            val_loss = self.validate()
            log.info(
                f"Epoch: {epoch+1}/{self.epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                timestmp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                model_file = f"{self.model_name}_{timestmp}_val_{val_loss:.4f}.pth"
                if val_loss < self.min_loss:
                    # Save only if its lower than our bare minimum to prevent useless files
                    self.save_model(model_file)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    log.info(
                        f"Early stopping triggered after {epoch+1} epochs with no improvement.")
                    break

    def validate(self):
        """Evaluate the model on the validation set and return the average loss"""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                if self.roi:
                    images = self._apply_roi_and_crop(images)
                else:
                    images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels.view(-1, 1).float())
                total_loss += loss.item()
        avg_val_loss = total_loss / len(self.val_loader)
        self.model.train()  # Return to train mode
        return avg_val_loss

    def test(self):
        """
        Load the model weights and evaluate the model on the test set 
        and return the average test loss.
        """
        log.info(f"Evaluating {self.model_name} model on the test set.")
        self.load_model()

        self.model.eval()
        total_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                if self.roi:
                    images = self._apply_roi_and_crop(images)
                else:
                    images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels.view(-1, 1).float())
                total_loss += loss.item()
                preds = torch.sigmoid(outputs).view(-1) > 0.5
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        avg_test_loss = total_loss / len(self.test_loader)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results = {
            "model_name": self.model_name,
            "weight_used": self.file_name,
            "test_loss": avg_test_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_path = os.path.join(
            results_dir, f"{self.model_name}_{timestamp}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        log.info(f"Test Loss: {avg_test_loss:.4f}")
        log.info(f"Test results saved at: {results_path}")
        return avg_test_loss

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
    else:
        log.error(f"{name} is not a valid model.")
        sys.exit(1)

    dimensions = [int(dim) for dim in CONSTANTS["models"][name].split("x")]

    return model, dimensions


if __name__ == "__main__":
    valid_models = ["mobilenetv3", "cnn", "efficientnet"]

    parser = argparse.ArgumentParser(
        description="Run models for training or evaluation")
    parser.add_argument('--models', nargs='+', required=True,
                        help='List of models to train, save, or evaluate')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'export'],
                        required=True, help='Mode of operation: train, evaluate for performance benchmark or export for mobile app.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for training')
    parser.add_argument('--batch', type=int, default=8,
                        help='Set the size of the batch for training and inference.')
    parser.add_argument('--fill_noise', action='store_true',
                        help="Fill missing area with Gaussian pixels or black", default=False)
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
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
    parser.add_argument("--min_loss", type=float,
                        help="Minimum loss needed to save the weights", default=0.5000)

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
    min_loss = args.min_loss
    roi = args.roi
    roi_weight = args.roi_weight
    fill_noise = args.fill_noise

    for model_name in list_of_models:
        model, dimensions = create_model_from_name(model_name)
        runner = Runner(model=model, lr=lr, epochs=epochs, is_loss_weighted=weighted_loss,
                        is_oversampled=weighted_sampling, batch_size=batch_size, patience=patience,
                        dimensions=dimensions, model_name=model_name, file_name=file_name,
                        min_loss=min_loss, roi=roi, roi_weight=roi_weight, fill_noise=fill_noise)

        if mode == "train":
            if not os.path.exists("dataset/Images"):
                log.error(
                    "Dataset does not exist for training, please download using data_setup.py before training.")
                sys.exit(1)
            runner.train()
        elif mode == "evaluate":
            runner.test()
        elif mode == "export":
            runner.export()

    if args.env == "cloud":
        auto_shutdown(args.copy_dir)
