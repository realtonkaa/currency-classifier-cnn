"""Model architecture for currency classification using MobileNetV2 transfer learning."""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class CurrencyClassifier(nn.Module):
    """CNN classifier using MobileNetV2 backbone with custom classification head.

    Uses pretrained MobileNetV2 features and replaces the final classifier
    layer to output predictions for our currency classes.
    """

    def __init__(self, num_classes: int = 112, dropout: float = 0.2):
        super().__init__()
        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Freeze early layers — only fine-tune later layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # Replace classifier head
        in_features = self.backbone.last_channel  # 1280 for MobileNetV2
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
