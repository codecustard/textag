"""Models. Step 4: a small CNN built from scratch (the baseline)."""

from __future__ import annotations

import torch.nn as nn

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


def build(name: str, **kw):
    return {"small_cnn": SmallCNN}[name](**kw)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
