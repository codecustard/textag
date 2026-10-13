"""Models: a small CNN built from scratch (baseline) and a pretrained ResNet18 (transfer learning)."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

NUM_CLASSES = 47


class SmallCNN(nn.Module):
    """Baseline: 4 conv blocks + 2 fully connected layers, trained from scratch.

    Each block: conv 3x3 -> batch norm -> ReLU -> max pool (halves the image size).
    224 -> 112 -> 56 -> 28 -> 14, then global average pool -> 256 numbers per image.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def resnet18(num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = True):
    """ResNet18 pretrained on ImageNet, with a new last layer for our classes.

    freeze_backbone=True: only the new last layer trains (feature extractor).
    Call unfreeze(model) later to fine-tune every layer.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze the pretrained layers so their weights don't change
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the last layer: 512 features -> 1000 ImageNet classes  becomes  512 -> our classes
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def unfreeze(model: nn.Module):
    """Make every layer trainable (for fine-tuning)."""
    for param in model.parameters():
        param.requires_grad = True


def build(name: str, **kw):
    return {"small_cnn": SmallCNN, "resnet18": resnet18}[name](**kw)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
