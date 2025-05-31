import os
import shutil
import sys
from zipfile import ZipFile

import gdown

from util.logger import setup_logger

logger = setup_logger()


def _download(data_dir: str, download_url: str):
    """Downloads the item from the url into the data directory.

    Args:
        data_dir (str): The directory to download into.
        download_url (str): The url to download from, assumes the artifact is a zip file.
    """
    # Create 'dataset' folder or clear it if it already exists
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    # Download the dataset zip file
    zip_path = os.path.join(os.getcwd(), "dataset.zip")
    gdown.download(
        download_url, zip_path, quiet=False)

    # Extract the zip file
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # Delete the zip file
    os.remove(zip_path)


def download_classification_dataset():
    """Download the dataset for the full classification with images and labels for classification.
    """

    data_dir = os.path.join(os.getcwd(), "dataset")
    _download(data_dir=data_dir,
              download_url="https://drive.google.com/uc?id=1I9XWVEN-aFm1_sdtvJsgKnJjUH2fS3Xg")


def download_yolo_dataset():
    """Download the dataset for yolo finetuning with images and bounding boxes for detection.
    """
    data_dir = os.path.join(os.getcwd(), "dataset", "roi")
    _download(data_dir=data_dir,
              download_url="https://drive.google.com/uc?id=1vMiqywMd-wlJOS8xYortG4aN69EHkNGZ")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("No download specified! Required: classification / yolo")

    dataset_type = sys.argv[1].lower()

    if dataset_type == "classification":
        logger.info("Downloading classification dataset.")
        download_classification_dataset()
    elif dataset_type == "yolo":
        logger.info("Downloading yolo dataset.")
        download_yolo_dataset()
    else:
        logger.error(
            "Incorrect download specified! Allowed: classification / yolo.")
