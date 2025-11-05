"""Image preprocessing and utility functions."""

from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision import transforms


def load_and_preprocess(
    image_path: Union[str, Path],
    img_size: int = 224,
) -> torch.Tensor:
    """Load a single image and preprocess for inference.

    Args:
        image_path: Path to the image file.
        img_size: Target size (height and width) for resizing.

    Returns:
        Preprocessed image tensor of shape (3, img_size, img_size).
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)


def get_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
