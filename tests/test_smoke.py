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


def test_predict_returns_top3_and_time():
    from PIL import Image
    from textag.infer import predict

    model = build("resnet18", pretrained=False).eval()
    classes = [f"class_{i}" for i in range(47)]
    preds, seconds = predict(model, classes, Image.new("RGB", (300, 300)))
    assert len(preds) == 3
    assert all(label in classes and 0 <= p <= 1 for label, p in preds)
    assert seconds > 0
