"""Tests for the CNN model architecture."""

import pytest
import torch
import torch.nn as nn

from src.model import CurrencyClassifier


def test_model_output_shape():
    model = CurrencyClassifier(num_classes=112)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 112)


def test_model_batch_output_shape():
    model = CurrencyClassifier(num_classes=50)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (4, 50)


def test_model_custom_num_classes():
    model = CurrencyClassifier(num_classes=10)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 10)


def test_model_trains_one_step():
    model = CurrencyClassifier(num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dummy_input = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.randint(0, 10, (4,))

    output = model(dummy_input)
    loss = criterion(output, dummy_labels)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0


def test_model_has_frozen_params():
    model = CurrencyClassifier(num_classes=10)
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert frozen > 0
    assert trainable > 0


def test_unfreeze_all():
    model = CurrencyClassifier(num_classes=10)
    model.unfreeze_all()
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    assert frozen == 0


def test_get_trainable_params():
    model = CurrencyClassifier(num_classes=10)
    count = model.get_trainable_params()
    assert count > 0
    assert isinstance(count, int)
