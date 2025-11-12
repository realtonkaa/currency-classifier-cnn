"""Tests for the dataset pipeline."""

import os
import pytest
import torch
from PIL import Image
from pathlib import Path

from src.dataset import BanknoteDataset, get_train_transforms, get_val_transforms, create_splits

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_data"


@pytest.fixture(autouse=True)
def create_sample_data():
    """Create small test images if they don't exist."""
    for currency in ["USD_1", "EUR_10"]:
        dir_path = FIXTURES_DIR / currency
        dir_path.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            img_path = dir_path / f"sample_{i}.jpg"
            if not img_path.exists():
                img = Image.new("RGB", (100, 50), color=(i * 80, 100, 150))
                img.save(img_path)
    yield


def test_load_dataset_returns_correct_shape():
    dataset = BanknoteDataset(root=str(FIXTURES_DIR), transform=get_val_transforms())
    img, label = dataset[0]
    assert img.shape == (3, 224, 224)
    assert isinstance(label, int)


def test_dataset_length():
    dataset = BanknoteDataset(root=str(FIXTURES_DIR))
    assert len(dataset) == 6  # 3 images * 2 classes


def test_dataset_classes():
    dataset = BanknoteDataset(root=str(FIXTURES_DIR))
    assert dataset.num_classes == 2
    assert "USD_1" in dataset.classes
    assert "EUR_10" in dataset.classes


def test_train_transforms_augment():
    transform = get_train_transforms()
    img = Image.new("RGB", (100, 50), color=(128, 128, 128))
    result = transform(img)
    assert result.shape == (3, 224, 224)


def test_val_transforms_no_augment():
    transform = get_val_transforms()
    img = Image.new("RGB", (100, 50), color=(128, 128, 128))
    result = transform(img)
    assert result.shape == (3, 224, 224)


def test_train_val_split():
    train_ds, val_ds = create_splits(root=str(FIXTURES_DIR), val_ratio=0.5)
    assert len(train_ds) + len(val_ds) == 6
    assert len(val_ds) >= 2  # At least some in validation


def test_dataset_returns_tensor_and_int():
    dataset = BanknoteDataset(root=str(FIXTURES_DIR), transform=get_val_transforms())
    for i in range(len(dataset)):
        img, label = dataset[i]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, int)
        assert 0 <= label < dataset.num_classes
