import argparse
import os
import sys

import requests
from tqdm import tqdm
from ultralytics import YOLO

from util.cloud_tools import auto_shutdown
from util.logger import logger

BASE_WEIGHT_PATH = "weights/yolo/yolov11s.pt"


def _download_yolo_baseweight():
    """Downloads the base weight for the YOLO model, to fine-tune further.
    """

    if not os.path.exists(BASE_WEIGHT_PATH):
        os.makedirs("weights/yolo")
        logger.info("Downloading base weight for YOLOv11s for finetuning.")
        weight_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
        response = requests.get(weight_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(BASE_WEIGHT_PATH, "wb") as f, tqdm(
            desc="Downloading weight",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        logger.info("YOLO weight download complete")
    else:
        logger.info("YOLO weight already exists, skipping download.")


class RoiRunner:
    def __init__(self, epochs: int, model: YOLO, file_name: str, batch_size: int) -> None:
        self.epochs = epochs
        self.model = model
        self.file_name = file_name
        self.batch_size = batch_size

    def train(self):
        self.model.train(model=BASE_WEIGHT_PATH, data="dataset/roi/data.yaml",
                         project="weights/yolo",
                         epochs=self.epochs, batch=self.batch_size)

    def evaluate(self):
        self.load_model(self.file_name)
        self.model.val(data="cfgs/yolo/data.yaml")

    def load_model(self, path: str):
        """Load the YOLO model weights from a specific path

        Args:
            path (str): The path to the model weight.
        """
        self.model = YOLO(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Region Of Interest Model with YoloV11s")
    parser.add_argument('--mode', choices=['train', 'evaluate'], type=str,
                        required=True, help="Mode of operation: Train or evaluate performance.")
    parser.add_argument('--batch', type=int, default=8,
                        help="Set the size of the batch for training.")
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument(
        '--file', type=str, help='File name of model weights to use when evaluating')
    parser.add_argument("--env", type=str, choices=["local", "cloud"], default="local",
                        help="Cloud mode has a special shutdown sequence to save resources.")
    parser.add_argument("--copy_dir", type=str,
                        help="Directory where logs and weights folder will be copied (required if env is cloud)")

    args = parser.parse_args()

    mode = args.mode
    epochs = args.epochs
    file_name = args.file if args.file else ""
    batch_size = args.batch

    # Validation of arguments
    if args.mode == 'evaluate':
        if not args.file:
            parser.error(
                '--file has to be specified when the mode is evaluate or export')
    if args.env == "cloud":
        if not args.copy_dir:
            parser.error("--copy_dir is required when env is cloud")
        elif not os.path.exists(args.copy_dir):
            logger.error(
                "The copy directory does not exist, halting execution.")
            sys.exit(1)

    # Base weights are needed no matter if training or evaluating
    _download_yolo_baseweight()

    model = YOLO(BASE_WEIGHT_PATH, task="detect")

    runner = RoiRunner(epochs=epochs, model=model, file_name=file_name,
                       batch_size=batch_size)

    if mode == "train":
        # Check if the dataset for RoI training exists before proceeding
        if not os.path.exists("dataset/roi"):
            logger.error(
                "Dataset does not exist for training, please download using data_setup.py before training.")
            sys.exit(1)
        runner.train()
    elif mode == "evaluate":
        runner.evaluate()

    if args.env == "cloud":
        auto_shutdown(args.copy_dir)
