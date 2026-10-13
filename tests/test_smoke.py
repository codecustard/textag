import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from textag.model import build  # noqa: E402


def test_small_cnn_outputs_47_classes():
    x = torch.randn(2, 3, 224, 224)
    assert build("small_cnn")(x).shape == (2, 47)


def test_resnet18_outputs_47_classes():
    x = torch.randn(2, 3, 224, 224)
    assert build("resnet18", pretrained=False)(x).shape == (2, 47)
