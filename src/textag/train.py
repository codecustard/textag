"""Training loop: one epoch, evaluation, and fit() with early stopping + LR scheduling."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():   # Apple GPU
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += loss_fn(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def fit(model, loaders, *, epochs=40, lr=1e-3, weight_decay=1e-4, patience=7,
        lr_patience=3, device=None, ckpt_path=None, arch=None, classes=None):
    """Train with AdamW + CrossEntropyLoss.

    - ReduceLROnPlateau: halve the learning rate when val loss stalls for `lr_patience` epochs.
    - Early stopping: stop when val loss hasn't improved for `patience` epochs,
      then restore the weights from the best epoch.
    Returns a history dict (one list per metric).
    """
    device = device or get_device()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=lr_patience)

    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "secs")}
    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device)
        va_loss, va_acc = evaluate(model, loaders["val"], loss_fn, device)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(va_loss)
        secs = time.perf_counter() - t0

        for k, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc, lr_now, secs)):
            hist[k].append(v)
        print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | lr {lr_now:.0e} | {secs:.0f}s", flush=True)

        if va_loss < best_loss:
            best_loss, best_epoch, bad = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                print(f"early stopping: no val-loss improvement for {patience} epochs")
                break

    model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch
    print(f"best epoch {best_epoch}: val loss {best_loss:.3f}, val acc {hist['val_acc'][best_epoch - 1]:.3f}")

    if ckpt_path:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": arch, "state_dict": model.state_dict(), "classes": classes}, ckpt_path)
    return hist
