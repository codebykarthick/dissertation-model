import os
import random

import matplotlib.pyplot as plt
import torch
from PIL import Image

PATH_TO_IMG = os.path.join(os.getcwd(), "..", "dataset", "Images")


def get_random_image(path=PATH_TO_IMG):
    """
    Returns a tuple (img_path, image), where:
     - img_path is the full filesystem path to a randomly chosen image
     - image is a PIL.Image opened in RGB mode
    """
    # List all image files in the directory
    image_files = [
        f for f in os.listdir(path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not image_files:
        raise FileNotFoundError(f"No images found in {path}")

    # Pick one at random
    chosen = random.choice(image_files)
    img_path = os.path.join(path, chosen)

    # Load and return
    image = Image.open(img_path).convert('RGB')
    return img_path, image


def show_pil_image(image, title=None):
    """
    Renders a PIL.Image in a Matplotlib figure.

    Args:
        image (PIL.Image): The image to display.
        title (str, optional): Title for the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image)        # Matplotlib understands PIL Images directly
    if title:
        plt.title(title)
    plt.axis('off')          # hide axes ticks
    plt.tight_layout()
    plt.show()


def yolo_model_crop_image(results, image: Image.Image) -> Image.Image:
    result = results[0]
    if not result.boxes:
        # no boxes detected → return full strip
        return image

    # pick the box with highest confidence
    boxes = result.boxes
    confidences = boxes.conf.cpu()
    best_idx = int(torch.argmax(confidences))
    best_conf = float(confidences[best_idx])

    # if top confidence is too low, skip cropping
    if best_conf < 0.3:
        return image

    # otherwise, crop the strip
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    return image.crop((x1, y1, x2, y2))


def display_tensor_as_image(tensor: torch.Tensor) -> None:
    """
    Display the cropped tensor as an image using matplotlib.
    Expects `cropped` to be a tensor of shape [1, C, H, W].
    """
    import matplotlib.pyplot as plt

    # Move to CPU, remove batch dim, convert to HWC numpy
    img_tensor = tensor.detach().cpu().squeeze(0)  # [C, H, W]
    img_np = img_tensor.permute(1, 2, 0).numpy()  # [H, W, C]

    # Clip values to [0,1] if necessary
    img_np = img_np.clip(0, 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(img_np)
    plt.axis('off')
    plt.show()
