import argparse
from datetime import datetime
from models.lfd_cnn import LFD_CNN
from models.pretrained_models import get_mobilenetv3, get_efficientnet
import os
import torch
from tqdm import tqdm
from util.constants import CONSTANTS
from util.data_loader import get_data_loaders
from util.logger import setup_logger
from util.screen_tools import get_weight_for_model

log = setup_logger()


class Runner:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else "cpu"
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders()
        self.criterion = torch.nn.BCELoss()
        learning_rate = CONSTANTS["learning_rate"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)

    def train(self, epochs=30):
        """
        Run the training loop for the loaded model for the specified epochs
        """
        self.model.to(self.device)
        self.model.train()

        best_val_loss = float('inf')
        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, labels in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
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
                f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")

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

    def load_model(self):
        """
        Load the model weights from a pth file in weights/ directory
        """
        model_filepath = get_weight_for_model(self.model_name)
        state_dict = torch.load(model_filepath, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        log.info(f"Model loaded from: {model_filepath}")


def create_model_from_name(name):
    """
    Create the model instance from the name provided in the arguments.
    """
    model = None
    if name == "cnn":
        model = LFD_CNN()
    elif name == "mobilenetv3":
        model = get_mobilenetv3()
    elif name == "efficientnet":
        model = get_efficientnet()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run models for training or evaluation")
    parser.add_argument('--models', nargs='+', required=True,
                        help='List of models to train, save, or evaluate')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'export'],
                        required=True, help='Mode of operation: train, evaluate for performance benchmark or export for mobile app.')
    args = parser.parse_args()

    list_of_models = args.models
    mode = args.mode

    for model_name in list_of_models:
        model = create_model_from_name(model_name)
        runner = Runner(model=model)

        if mode == "train":
            runner.train()
        elif mode == "evaluate":
            runner.test()
        elif mode == "export":
            runner.export()
