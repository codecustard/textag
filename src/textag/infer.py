"""Inference: load the trained model once, then predict the top-k classes for one image, timed."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image

from textag.data import eval_tf
from textag.model import build


def load_model(path: str | Path):
    """Load a checkpoint saved as {"arch", "state_dict", "classes"}. Returns (model, classes) on CPU."""
    checkpoint = torch.load(path, map_location="cpu")
    classes = checkpoint["classes"]
    kw = {"pretrained": False, "freeze_backbone": False} if checkpoint["arch"] == "resnet18" else {}
    model = build(checkpoint["arch"], num_classes=len(classes), **kw)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes


def predict(model, classes, image, top_k: int = 3):
    """Predict one image (file path or PIL image).

    Returns ([(label, probability), ...], seconds). The time covers preprocessing + model + softmax,
    i.e. what a user waits for. Model loading is not included (done once at startup).
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    start = time.perf_counter()
    x = eval_tf(image).unsqueeze(0)           # add a batch dimension: (1, 3, 224, 224)
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1)[0]
    top_probs, top_idx = probs.topk(top_k)
    seconds = time.perf_counter() - start

    return [(classes[i], p.item()) for p, i in zip(top_probs, top_idx)], seconds
