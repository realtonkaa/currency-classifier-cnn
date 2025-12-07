"""Tests for the prediction module."""

import pytest
import torch
from pathlib import Path
from PIL import Image

from src.predict import predict_currency, load_model, get_top_k, CURRENCY_NAMES


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_image(tmp_path):
    """Create a dummy test image."""
    img_path = tmp_path / "test_bill.jpg"
    img = Image.new("RGB", (200, 100), color=(100, 150, 200))
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def dummy_model(tmp_path):
    """Create and save a dummy model for testing."""
    from src.model import CurrencyClassifier

    model = CurrencyClassifier(num_classes=5)
    model_path = tmp_path / "test_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "val_acc": 50.0,
    }, model_path)
    return str(model_path)


def test_load_model(dummy_model):
    model = load_model(dummy_model, num_classes=5)
    assert model is not None
    assert not model.training  # Should be in eval mode


def test_predict_returns_expected_keys(sample_image, dummy_model):
    class_names = ["USD_1", "USD_5", "EUR_10", "GBP_20", "INR_500"]
    result = predict_currency(
        sample_image,
        model_path=dummy_model,
        class_names=class_names,
        num_classes=5
    )
    assert "currency_code" in result
    assert "currency_name" in result
    assert "denomination" in result
    assert "confidence" in result
    assert "top_3" in result
    assert 0 <= result["confidence"] <= 1


def test_predict_top3_has_correct_structure(sample_image, dummy_model):
    class_names = ["USD_1", "USD_5", "EUR_10", "GBP_20", "INR_500"]
    result = predict_currency(
        sample_image,
        model_path=dummy_model,
        class_names=class_names,
        num_classes=5
    )
    assert len(result["top_3"]) == 3
    for pred in result["top_3"]:
        assert "currency_code" in pred
        assert "denomination" in pred
        assert "confidence" in pred


def test_get_top_k():
    probs = torch.tensor([[0.5, 0.3, 0.1, 0.05, 0.05]])
    class_names = ["USD_1", "EUR_10", "GBP_5", "INR_100", "JPY_1000"]
    results = get_top_k(probs, class_names, k=3)
    assert len(results) == 3
    assert results[0]["confidence"] >= results[1]["confidence"]


def test_currency_names_mapping():
    assert CURRENCY_NAMES["USD"] == "US Dollar"
    assert CURRENCY_NAMES["EUR"] == "Euro"
    assert len(CURRENCY_NAMES) > 10
