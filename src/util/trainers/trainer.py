import glob
import json
import os
import sys
from datetime import datetime

import torch

from models.classification.lfd_cnn import LFD_CNN, KDStudent
from models.classification.pretrained_models import (
    get_efficientnet_tuned,
    get_mobilenetv3_tuned,
    get_shufflenet_tuned,
    get_tinyvit_tuned,
)
from models.siamese.efficientnet import SiameseEfficientNet
from models.siamese.mobilenet import SiameseMobileNet
from models.siamese.shufflenet import SiameseShuffleNet
from util.constants import CONSTANTS
from util.logger import setup_logger


class Trainer:
    """The base class from which all trainers should inherit to implement their own logic.
    """

    def __init__(self, roi: bool, fill_noise: bool, model_name: str,
                 num_workers: int, k: int, is_sampling_weighted: bool, is_loss_weighted: bool,
                 batch_size: int, epochs: int, task_type: str, lr: float, patience: int,
                 label: str, roi_weight: str = "", delta=0.02, filename: str = ""):
        self.log = setup_logger()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.fill_noise = fill_noise
        self.num_workers = num_workers
        self.k = k
        self.is_sampling_weighted = is_sampling_weighted
        self.is_loss_weighted = is_loss_weighted
        self.batch_size = batch_size

        self.epochs = epochs
        self.task_type = task_type
        self.lr = lr
        self.patience = patience
        self.delta = delta
        self.label = label
        self.roi = roi
        self.roi_model = None
        self.filename = filename
        self.roi_weight = roi_weight

    def create_model_from_name(self, name: str, task_type: str) -> tuple[torch.nn.Module, list[int]]:
        """Create the model instance from the name of the model specified. Halts execution
        if model name is wrong.

        Args:
            name (str): Name of the model to be used.
            task_type (str): Type of model — classification or siamese

        Returns:
            tuple[torch.nn.Module, list[int]]: Returns the instance of the model along with the dimensions to be used.
        """
        if "classification" in task_type:
            if name == "cnn":
                model = LFD_CNN()
            elif name == "mobilenetv3":
                model = get_mobilenetv3_tuned()
            elif name == "efficientnet":
                model = get_efficientnet_tuned()
            elif name == "shufflenet":
                model = get_shufflenet_tuned()
            elif name == "tinyvit":
                model = get_tinyvit_tuned()
            else:
                self.log.error(f"{name} is not a valid model.")
                sys.exit(1)
        elif task_type == "siamese":
            if name == "mobilenetv3":
                model = SiameseMobileNet()
            elif name == "efficientnet":
                model = SiameseEfficientNet()
            elif name == "shufflenet":
                model = SiameseShuffleNet()
            else:
                self.log.error(f"{name} is not a valid model.")
                sys.exit(1)
        elif task_type == "distillation":
            if name == "student":
                model = KDStudent()
        elif task_type == "gradcam":
            if name == "efficientnet":
                model = get_efficientnet_tuned()
            elif name == "shufflenet":
                model = get_shufflenet_tuned()
            elif name == "student":
                model = KDStudent()
        else:
            self.log.error(f"{task_type} is not a valid type.")
            sys.exit(1)

        model = model.to(self.device)
        dimensions = [int(dim) for dim in CONSTANTS["models"][name].split("x")]

        return model, dimensions

    def train(self):
        raise NotImplementedError("Train method must be overriden!")

    def evaluate(self):
        raise NotImplementedError("Evaluate method must be overriden!")

    def export(self):
        raise NotImplementedError("Export method must be overriden!")

    def _cleanup_old_files(self, pattern: str, retain_last: int = 2):
        if retain_last > 0:
            all_files = sorted(
                glob.glob(pattern),
                key=os.path.getmtime,
                reverse=True
            )

            if not all_files:
                # No files to clean up
                return

            self.log.info(
                f"Found {len(all_files) - retain_last} files to clean up in folder.")

            for old_file in all_files[retain_last:]:
                try:
                    os.remove(old_file)
                except Exception as e:
                    self.log.warning(f"Failed to delete {old_file}: {e}")

    def evaluate_and_save(self, current_metric: float, best_metric: float, model: torch.nn.Module,
                          metrics: dict[str, float], fold: int = 0, save_model: bool = True, retain_last: int = 2) -> bool:
        """Evaluate if it satifies conditions before saving the model and metrics results.

        Args:
            current_metric (float): The value of the metric being evaluated for the current epoch.
            best_metric (float): The best value of the metric so far.
            model (torch.nn.Module): The model to save.
            metrics (dict[str, float]): The metrics to save.
            fold (int): The fold for naming.

        Returns:
            bool: Returns True if it was saved, False otherwise for early stopping.
        """
        if current_metric > best_metric and current_metric > 0 and (current_metric - best_metric) > self.delta:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            model_file = f"{self.model_name}_fold{fold+1}_{timestamp}.pth"

            if save_model:
                self.save_model(model, model_file)

            filename = f"{self.model_name}_fold{fold+1}_{timestamp}_metrics.json"
            self.save_results(metrics=metrics, filename=filename)
            self.log.info(
                f"Saved metrics for Fold {fold + 1} in: {filename}")
            self._cleanup_old_files(
                pattern=f"weights/{self.label}/{self.model_name}/{self.model_name}_fold{fold+1}*.pth", retain_last=retain_last)
            self._cleanup_old_files(
                pattern=f"results/{self.label}/{self.model_name}_fold{fold+1}*.json", retain_last=retain_last)

            return True
        return False

    def save_model(self, model: torch.nn.Module, filename: str = "sample.pth"):
        """Save the model weights as a pth file in weights/ directory

        Args:
            filename (str, optional): Filename to save the weights as. Defaults to "sample.pth".
        """
        model_weights_dir = os.path.join(
            os.getcwd(), CONSTANTS["weights_path"], self.label, self.model_name)

        if not os.path.exists(model_weights_dir):
            self.log.info(f"Creating folder at: {model_weights_dir}")
            os.makedirs(model_weights_dir, exist_ok=True)

        model_filepath = os.path.join(model_weights_dir, filename)

        torch.save(model.state_dict(), model_filepath)
        self.log.info(f"Model saved in: {filename}")

    def save_results(self, metrics: dict[str, float], filename="sample.json"):
        """Save the result metrics as a json file

        Args:
            filename (str, optional): The filename to save as. Defaults to "sample.json".
        """
        results_dir = os.path.join(os.getcwd(), "results", self.label)
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(
            results_dir, filename)
        with open(result_file, "w") as f:
            json.dump(metrics, f, indent=4)

    def load_model(self, model: torch.nn.Module, filename: str = "sample.pth"):
        model_weights_dir = os.path.join(
            os.getcwd(), CONSTANTS["weights_path"], self.label, self.model_name)

        model_filepath = os.path.join(model_weights_dir, filename)

        model.load_state_dict(torch.load(model_filepath))
        self.log.info(f"Model loaded from: {model_filepath}")
