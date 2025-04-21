import gdown
import os
import shutil
from zipfile import ZipFile

DATASET_URL = "https://drive.google.com/file/d/1YKLtYfpc66mBaEUh84QYmGKvDYYbzXxK/view?usp=share_link"

if __name__ == "__main__":
    # Create 'dataset' folder or clear it if it already exists
    data_dir = os.path.join(os.getcwd(), "dataset")
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    # Download the dataset zip file
    zip_path = os.path.join(os.getcwd(), "dataset.zip")
    gdown.download(
        "https://drive.google.com/uc?id=1YKLtYfpc66mBaEUh84QYmGKvDYYbzXxK", zip_path, quiet=False)

    # Extract the zip file
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())

    # Delete the zip file
    os.remove(zip_path)
