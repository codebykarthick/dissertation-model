import os

from PIL import Image
from tqdm import tqdm

IMAGE_DIM = (512, 512)
DATASET_PATH = "dataset/Images"
OUTPUT_PATH = "dataset/roi"


def resize_with_padding(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
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


if __name__ == "__main__":
    """Program to resize to the maximum dimensions of 256x256 to ensure annotation is constant.
    """
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for img_name in tqdm(os.listdir(DATASET_PATH), desc="Preprocessing Progress"):
        img_path = os.path.join(DATASET_PATH, img_name)

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")  # Ensures compatibility
                resized_img = resize_with_padding(img, IMAGE_DIM)
                resized_img.save(os.path.join(OUTPUT_PATH, img_name))
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")
