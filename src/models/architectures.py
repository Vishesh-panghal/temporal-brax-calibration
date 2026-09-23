"""Model architectures for BRAX thoracic disease classification."""

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


def build_model(
    architecture: str = "densenet121",
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Builds a vision backbone for multi-label chest radiography classification.
    
    Supports:
        - 'densenet121': CheXNet standard architecture
        - 'resnet50': Deep residual network benchmark
    """
    arch = architecture.lower().replace("-", "")

    if arch == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported architecture: '{architecture}'. Choose 'densenet121' or 'resnet50'.")

    return model
