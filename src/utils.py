"""Image preprocessing and utility functions."""

from PIL import Image
from torchvision import transforms


def load_and_preprocess(image_path: str, img_size: int = 224):
    """Load a single image and preprocess for inference."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)
