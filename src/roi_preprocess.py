import os
import random
import shutil
import sys

from PIL import Image
from tqdm import tqdm

IMAGE_DIM = (512, 512)
DATASET_PATH = "dataset/Images"
OUTPUT_PATH = "dataset/roi"
SPLIT_RATIO = 0.9


def _resize_with_padding(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resize the image maintaining aspect ratio and pad missing pixels with black.

    Args:
        image (Image.Image): The image to resize
        target_size (tuple[int, int]): The target dimensions to resize to.

    Returns:
        Image.Image: The resized image to be used for the final annotation.
    """
    original_size = image.size
    ratio = min(target_size[0] / original_size[0],
                target_size[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))

    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Create a new image with black background
    new_image = Image.new("RGB", target_size, (0, 0, 0))

    # Center the resized image
    paste_position = (
        (target_size[0] - new_size[0]) // 2,
        (target_size[1] - new_size[1]) // 2
    )
    new_image.paste(image, paste_position)

    return new_image


def _move_pair(filename_list: list[str], dest_folder: str) -> None:
    """Move the pair of image and its annotation to the destination folder specified.

    Args:
        filename_list (list[str]): The list of files to be moved.
        dest_folder (str): The destination folder to move the files to.
    """
    for filename in filename_list:
        png_name = filename + ".png"
        txt_name = filename + ".txt"

        # Source paths:
        src_png = os.path.join(OUTPUT_PATH, png_name)
        src_txt = os.path.join(OUTPUT_PATH, txt_name)

        # Destination paths:
        dst_png = os.path.join(dest_folder, png_name)
        dst_txt = os.path.join(dest_folder, txt_name)

        # Move .png
        if os.path.exists(src_png):
            shutil.move(src_png, dst_png)
        else:
            print(f"Warning: '{src_png}' not found")

        # Move .txt
        if os.path.exists(src_txt):
            shutil.move(src_txt, dst_txt)
        else:
            print(f"Warning: '{src_txt}' not found")


if __name__ == "__main__":
    """Program to resize to the maximum dimensions of 256x256 to ensure annotation is constant. It also splits the data
    into train and validation sets for actual training.
    """

    if len(sys.argv) < 2:
        print(f"No args sent for processing! Allowed: resize / split")

    mode = sys.argv[1]

    if mode == "resize":
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        for img_name in tqdm(os.listdir(DATASET_PATH), desc="Preprocessing Progress"):
            img_path = os.path.join(DATASET_PATH, img_name)

            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")  # Ensures compatibility
                    resized_img = _resize_with_padding(img, IMAGE_DIM)
                    resized_img.save(os.path.join(OUTPUT_PATH, img_name))
            except Exception as e:
                print(f"Failed to process {img_name}: {e}")
    elif mode == "split":
        train_folder = os.path.join(OUTPUT_PATH, "train")
        val_folder = os.path.join(OUTPUT_PATH, "val")

        # Get it without extension and eliminate duplicates.
        files = list(set([file.split(".")[0]
                     for file in os.listdir(OUTPUT_PATH)]))
        print(f"Processing {len(files)} files.")

        if not os.path.exists(train_folder):
            os.makedirs(train_folder)

        if not os.path.exists(val_folder):
            os.makedirs(val_folder)

        # Shuffle so that its randomised
        random.shuffle(files)
        split_index = int(len(files) * SPLIT_RATIO)

        train_files = files[:split_index]
        val_files = files[split_index:]

        _move_pair(train_files, train_folder)
        _move_pair(val_files, val_folder)
        print(
            f"Split complete. {len(train_files)} images → '{train_folder}/', {len(val_files)} images → '{val_folder}/'.")
