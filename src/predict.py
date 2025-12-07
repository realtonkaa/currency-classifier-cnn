"""Prediction module for currency classification."""

import torch
from pathlib import Path
from typing import Dict, List, Optional

from src.model import CurrencyClassifier
from src.utils import load_and_preprocess


# Currency display names
CURRENCY_NAMES = {
    "USD": "US Dollar",
    "EUR": "Euro",
    "GBP": "British Pound",
    "INR": "Indian Rupee",
    "JPY": "Japanese Yen",
    "CAD": "Canadian Dollar",
    "AUD": "Australian Dollar",
    "CHF": "Swiss Franc",
    "CNY": "Chinese Yuan",
    "BRL": "Brazilian Real",
    "MXN": "Mexican Peso",
    "KRW": "South Korean Won",
    "SGD": "Singapore Dollar",
    "HKD": "Hong Kong Dollar",
    "SEK": "Swedish Krona",
    "NOK": "Norwegian Krone",
    "TRY": "Turkish Lira",
}


def load_model(model_path: str, num_classes: int, device: str = "cpu") -> CurrencyClassifier:
    """Load a trained model from checkpoint."""
    model = CurrencyClassifier(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)
    return model


def get_top_k(probabilities: torch.Tensor, class_names: List[str], k: int = 3) -> List[Dict]:
    """Get top-k predictions with probabilities."""
    topk_probs, topk_indices = probabilities.topk(k, dim=1)

    results = []
    for prob, idx in zip(topk_probs[0], topk_indices[0]):
        label = class_names[idx.item()]
        currency_code, denomination = label.split("_", 1)
        results.append({
            "currency_code": currency_code,
            "currency_name": CURRENCY_NAMES.get(currency_code, currency_code),
            "denomination": denomination,
            "confidence": round(prob.item(), 4)
        })
    return results


def predict_currency(image_path: str, model_path: str = "models/best_model.pth",
                     class_names: Optional[List[str]] = None,
                     num_classes: int = 112, device: str = "cpu") -> Dict:
    """Predict currency and denomination from a banknote image.

    Args:
        image_path: Path to the banknote image
        model_path: Path to the trained model checkpoint
        class_names: List of class names (e.g., ["EUR_10", "USD_1", ...])
        num_classes: Number of classes the model was trained on
        device: Device to run inference on

    Returns:
        dict with currency, denomination, confidence, and top-3 predictions
    """
    # Load model
    model = load_model(model_path, num_classes, device)

    # Preprocess image
    image = load_and_preprocess(image_path)
    image = image.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = probabilities.max(1)

    # Use class names if provided, otherwise generate generic ones
    if class_names is None:
        class_names = [f"CLASS_{i}" for i in range(num_classes)]

    predicted_label = class_names[predicted_idx.item()]

    # Parse label (format: "CURRENCY_DENOMINATION")
    if "_" in predicted_label:
        currency_code, denomination = predicted_label.split("_", 1)
    else:
        currency_code = predicted_label
        denomination = "unknown"

    return {
        "currency_code": currency_code,
        "currency_name": CURRENCY_NAMES.get(currency_code, currency_code),
        "denomination": denomination,
        "confidence": round(confidence.item(), 4),
        "top_3": get_top_k(probabilities, class_names, k=3)
    }
