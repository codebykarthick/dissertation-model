import argparse
from datetime import datetime
from models.lfd_cnn import LFD_CNN
from models.pretrained_models import get_mobilenetv3, get_efficientnet
import os
import torch
from tqdm import tqdm
from util.constants import CONSTANTS
from util.cloud_tools import auto_shutdown
from util.data_loader import get_data_loaders
from util.logger import setup_logger
import sys

log = setup_logger()


class Runner:
    def __init__(self, model: torch.nn.Module, model_name: str, lr: float, epochs: int,
                 dimensions: list[int, int], is_loss_weighted: bool, is_oversampled: bool,
                 file_name: str, batch_size: int):
        self.model = model
        self.model_name = model_name
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else "cpu"

        # Dimensions is passed here so that we get the appropriate images for training
        # TODO: Fix the additional args required
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            dimensions=dimensions, is_sampling_weighted=is_oversampled, batch_size=batch_size)

        if is_loss_weighted:
            # TODO: Fix weighted loss
            pos_weight = torch.tensor([5.0], device=self.device)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.file_name = file_name

    def train(self):
        """
        Run the training loop for the loaded model for the specified epochs
        """
        log.info(
            f"Training {self.model_name} model, for {self.epochs} epochs.")
        self.model.to(self.device)
        self.model.train()

        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for images, labels in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}', leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                # Ensure labels are float for BCELoss
                loss = self.criterion(outputs, labels.float())
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
                timestmp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                model_file = f"{self.model_name}_{timestmp}_val_{val_loss:.4f}.pth"
                self.save_model(model_file)

    def validate(self):
        """Evaluate the model on the validation set and return the average loss"""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels.float())
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
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels.float())
                total_loss += loss.item()
        avg_test_loss = total_loss / len(self.test_loader)
        log.info(f"Test Loss: {avg_test_loss:.4f}")
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
            os.makedirs(model_weights_dir, exist_ok=True)

        model_filepath = os.path.join(model_weights_dir, filename)

        torch.save(self.model.state_dict(), model_filepath)
        log.info(f"Model saved at: {model_filepath}")

    def get_model_filepath(self) -> str:
        model_weights_path = os.path.join(
            os.getcwd(), CONSTANTS['weights_path'], self.model_name, self.file_name)

        if not os.path.exists(model_weights_path):
            log.error(
                'Weights path does not exist to load model (probably no training happened).')
            sys.exit(1)

    def load_model(self):
        """
        Load the model weights from a pth file in weights/ directory
        """
        model_filepath = self.get_model_filepath()
        state_dict = torch.load(model_filepath, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        log.info(f"Model loaded from: {model_filepath}")


def create_model_from_name(name):
    """
    Create the model instance from the name provided in the arguments. Also
    returns the dimensions required for resizing.
    """

    if name == "cnn":
        model = LFD_CNN()
    elif name == "mobilenetv3":
        model = get_mobilenetv3()
    elif name == "efficientnet":
        model = get_efficientnet()

    dimensions = [int(dim) for dim in CONSTANTS["models"][name].split("x")]

    return model, dimensions


if __name__ == "__main__":
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
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument(
        '--file', type=str, help='File name of model weights to use when evaluating')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='Enable weighted loss to handle class imbalance')
    parser.add_argument('--weighted_sampling', action='store_true',
                        help='Enable weighted sampling for training data')
    parser.add_argument("--env", type=str, choices=["local", "cloud"], default="local",
                        help="Cloud mode has a special shutdown sequence to save resources.")
    parser.add_argument("--copy_dir", type=str,
                        help="Directory where logs and weights folder will be copied (required if env is cloud)")

    args = parser.parse_args()

    # Manual validation of the supplied arguments
    if args.mode == 'evaluate' or args.mode == 'export':
        if not args.file:
            parser.error(
                '--file has to be specified when the mode is evaluate or export')
    if args.weighted_loss == True:
        if args.weighted_sampling == True:
            parser.error(
                'Either one of weighted loss or weighted sampling can be set True at a time.')
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
    file_name = args.file
    batch_size = args.batch

    for model_name in list_of_models:
        model, dimensions = create_model_from_name(model_name)
        runner = Runner(model=model, lr=lr, epochs=epochs,
                        dimensions=dimensions, is_loss_weighted=weighted_loss,
                        is_oversampled=weighted_sampling, batch_size=batch_size)

        if mode == "train":
            runner.train()
        elif mode == "evaluate":
            runner.test()
        elif mode == "export":
            runner.export()

    if args.env == "cloud":
        auto_shutdown(args.copy_dir)
