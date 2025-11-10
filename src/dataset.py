"""Data loading and augmentation pipeline for banknote classification."""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image


# Class names mapping — covers major world currencies
# Format: "CURRENCY_DENOMINATION" e.g., "USD_20", "EUR_50", "INR_500"
CLASS_NAMES = []  # Populated dynamically from directory structure


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Return augmentation transforms for training.

    Applies random flips, rotations, colour jitter, and affine shifts to
    help the model generalise to varying photo conditions (lighting, angle,
    slight blur).

    Args:
        img_size: Resize target (height == width).

    Returns:
        A torchvision Compose pipeline ready to pass to BanknoteDataset.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Return deterministic transforms for validation and inference.

    No random augmentation — just resize and normalise so results are
    reproducible across evaluation runs.

    Args:
        img_size: Resize target (height == width).

    Returns:
        A torchvision Compose pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


class BanknoteDataset(Dataset):
    """Custom dataset for banknote images.

    Expects directory structure:
        root/
            USD_1/
                img1.jpg
                img2.jpg
            USD_5/
                ...
            EUR_10/
                ...
    """

    def __init__(self, root: str, transform: Optional[transforms.Compose] = None):
        self.root = Path(root)
        self.transform = transform or get_val_transforms()

        # Build class list from subdirectories
        self.classes = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all image paths and labels
        self.samples = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def create_splits(root: str, val_ratio: float = 0.2,
                  seed: int = 42) -> Tuple[Dataset, Dataset]:
    """Split a BanknoteDataset into train and validation subsets.

    Args:
        root: Root directory of the dataset (same structure as BanknoteDataset).
        val_ratio: Fraction of samples to use for validation (default 0.2 = 20%).
        seed: Random seed for reproducible splits.

    Returns:
        A (train_dataset, val_dataset) tuple. Each subset has the appropriate
        transforms applied: augmentation for train, plain resize for val.
    """
    full_dataset = BanknoteDataset(root, transform=None)

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Apply different transforms to each split
    train_ds.dataset.transform = get_train_transforms()
    val_ds.dataset.transform = get_val_transforms()

    return train_ds, val_ds
