import os
import random
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor

PATH_TO_IMG = os.path.join(os.getcwd(), "..", "dataset", "Images")


class ApplyCLAHE:
    """Applies Contrast Equalisation onto Photos for better contrast.
    """

    def __call__(self, img):
        img_np = np.array(img)
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return Image.fromarray(img_clahe)


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


def get_yolo_transform() -> transforms.Compose:
    """
    Returns a torchvision transform pipeline for YOLO inference:
    - Only creates a tensor needed for resize and cropping
    """
    return transforms.Compose([
        transforms.ToTensor()
    ])


def optimize_and_save(model: Union[torch.nn.Module, torch.jit.ScriptModule],
                      filepath: str,
                      freeze: bool = False,
                      save: bool = False) -> torch.jit.ScriptModule:
    """Optimises the model and saves it in the filepath specified

    Args:
        model (_type_): _description_
        path (str): _description_
        filename (str): _description_
        freeze (bool, optional): _description_. Defaults to False.
        save (bool, optional): _description_. Defaults to False.
    """
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

    if not isinstance(model, torch.jit.ScriptModule):
        model = torch.jit.script(model)

    if freeze == True:
        model = model.eval()
        model = torch.jit.freeze(model)

    model = optimize_for_mobile(model)
    if save == True:
        torch.jit.save(model, filepath)

    return model


def resize_and_tensor(tensor: torch.Tensor, img_size: int = 224) -> torch.Tensor:
    """
    Resizes the tensor to the given img_size using LANCZOS resampling,
    and converts it back to a tensor.
    Assumes input tensor is [1, C, H, W] in range [0, 1].
    """
    pil_image = to_pil_image(tensor.squeeze(0))  # [C, H, W] → PIL
    pil_image.thumbnail(
        (img_size, img_size), Image.Resampling.LANCZOS)
    # Create black background and paste resized image centered
    new_img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    left = (img_size - pil_image.size[0]) // 2
    top = (img_size - pil_image.size[1]) // 2
    new_img.paste(pil_image, (left, top))
    new_img = ApplyCLAHE()(new_img)
    # Convert back to tensor
    return to_tensor(new_img).unsqueeze(0)
