from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "mobilenet_v2"  # or "efficientnet_b0"
    num_classes: int = 7


class EmotionNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.model_type == "mobilenet_v2":
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, cfg.num_classes))
            self.backbone = base
        elif cfg.model_type == "efficientnet_b0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, cfg.num_classes))
            self.backbone = base
        else:
            raise ValueError("model_type must be 'mobilenet_v2' or 'efficientnet_b0'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_emotion_model(
    *,
    weights_path: str | None,
    cfg: ModelConfig,
    device: str | torch.device = "cpu",
) -> EmotionNet:
    model = EmotionNet(cfg)

    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state, strict=True)

    model.eval()
    model.to(device)
    return model

