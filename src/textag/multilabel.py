"""Multi-label tags and search: each image can have several of the 47 DTD tags (e.g. striped + woven).

DTD ships a second label file, labels_joint_anno.txt, with every tag people gave each image.
Most images have 1 tag; about 16% have 2-4. Same images, splits and transforms as data.py;
only the labels change:

  single-label: y = 7                          (one class number)
  multi-label:  y = [0, 0, 1, 0, ..., 1, 0]    (47 yes/no values, 1 = has this tag)

Training uses BCEWithLogitsLoss: a sigmoid per tag, so each tag is its own yes/no
(softmax makes the 47 scores compete and add up to 100%; sigmoids don't).

train_step / val_step / train: same as engine.py, except the accuracy.
engine.py checks "highest score == the one label"; here an image counts as correct when its
highest-scoring tag is one of its tags ("top tag correct"). With 1 tag per image this is the same thing.

index:   run every image through a model once, keep its tag scores and its embedding (for search)
metrics: precision@k for tag search and find-similar search
"""
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from textag import data
from textag.model import SmallCNN

LABELS_FILE = data.DATA_DIR / "dtd" / "dtd" / "labels" / "labels_joint_anno.txt"


# ---------------------------------------------------------------- data

def read_tags() -> Dict[str, List[str]]:
    """{"banded/banded_0002.jpg": ["banded"], "woven/woven_0115.jpg": ["woven", "striped"], ...}"""
    tags = {}
    for line in LABELS_FILE.read_text().splitlines():
        parts = line.split()
        if parts:
            tags[parts[0]] = parts[1:]
    return tags


class MultiLabelDTD(Dataset):
    """The DTD split from data.py, but y is 47 yes/no values instead of one class number."""

    def __init__(self, split: str, partition: int = 1):
        self.dtd = data.dataset(split, partition)      # same images and transforms as before
        self.classes = self.dtd.classes
        self.files = self.dtd._image_files             # image paths, in dataset order
        tags = read_tags()

        self.targets = []
        for path in self.files:
            key = f"{path.parent.name}/{path.name}"     # e.g. "woven/woven_0115.jpg"
            target = torch.zeros(len(self.classes))
            for tag in tags[key]:
                target[self.classes.index(tag)] = 1
            self.targets.append(target)
        self.targets = torch.stack(self.targets)        # (images, 47)

    def __len__(self):
        return len(self.dtd)

    def __getitem__(self, i):
        image, _ = self.dtd[i]                          # ignore the single label
        return image, self.targets[i]


def loaders(batch_size: int = 32, num_workers: int = 2, partition: int = 1):
    """train (shuffled) / val / test DataLoaders with multi-label targets."""
    ds = {s: MultiLabelDTD(s, partition) for s in ("train", "val", "test")}
    return {
        s: DataLoader(d, batch_size=batch_size, shuffle=(s == "train"), num_workers=num_workers)
        for s, d in ds.items()
    }


# ---------------------------------------------------------------- training

def top_tag_correct(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Share of images whose highest-scoring tag is one of their tags."""
    top_tag = logits.argmax(dim=1)                              # (batch,)
    hits = targets[torch.arange(len(targets)), top_tag]         # 1 if that tag is a true tag
    return hits.mean().item()


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a model for one epoch. Returns (train_loss, train_acc)."""
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Accuracy: top tag correct
        train_acc += top_tag_correct(y_pred, y)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Evaluates a model on the validation data for one epoch. Returns (val_loss, val_acc)."""
    model.eval()
    val_loss, val_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            val_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            # Accuracy: top tag correct
            val_acc += top_tag_correct(val_pred_logits, y)

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          scheduler=None,
          patience: int = None,
          verbose: bool = True) -> Dict[str, List]:
    """Same as engine.train (LR scheduler, early stopping, keeps the best weights), with the steps above."""
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Early stopping bookkeeping
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    epochs_without_improvement = 0

    model.to(device)

    for epoch in tqdm(range(epochs), disable=not verbose):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        val_loss, val_acc = val_step(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=loss_fn,
                                     device=device)

        if verbose:
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.4f} | "
                f"lr: {optimizer.param_groups[0]['lr']:.0e}"
            )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # LR scheduler: halve the learning rate if val loss hasn't improved for a few epochs
        if scheduler is not None:
            scheduler.step(val_loss)

        # Early stopping: remember the best weights; stop if no improvement for `patience` epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping: no val_loss improvement for {patience} epochs")
                break

    # Put the best weights back into the model
    model.load_state_dict(best_weights)
    results["best_epoch"] = best_epoch
    if verbose:
        print(f"Best epoch: {best_epoch} | val_loss: {best_val_loss:.4f} | "
              f"val_acc: {results['val_acc'][best_epoch - 1]:.4f}")
    return results


# ---------------------------------------------------------------- search index

def embed(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """The numbers just before the last layer: one "description" of each image.

    SmallCNN: the 256 numbers after global average pooling.
    ResNet18: the 512 numbers that go into its last layer (fc).
    """
    if isinstance(model, SmallCNN):
        return model.features(X).flatten(1)                         # (batch, 256)
    body = nn.Sequential(*list(model.children())[:-1])              # every ResNet layer except fc
    return body(X).flatten(1)                                       # (batch, 512)


def index(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """Run every image through the model once. Returns (tag_scores, embeddings), both on the CPU.

    tag_scores: (images, 47), sigmoid of each tag, 0-1
    embeddings: (images, 256 or 512), scaled to length 1 so `a @ b.T` gives similarity scores
    """
    model.to(device)
    model.eval()
    all_scores, all_embeddings = [], []
    with torch.inference_mode():
        for X, _ in dataloader:
            X = X.to(device)
            all_scores.append(torch.sigmoid(model(X)).cpu())
            all_embeddings.append(embed(model, X).cpu())
    embeddings = torch.cat(all_embeddings)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return torch.cat(all_scores), embeddings


# ---------------------------------------------------------------- search metrics

def tag_search_precision(tag_scores: torch.Tensor, targets: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Search each tag: rank all images by that tag's score, take the top k.

    Returns, per tag, the share of those k images that really have the tag. Shape (47,).
    """
    precisions = []
    for tag in range(targets.shape[1]):
        top_k = tag_scores[:, tag].topk(k).indices
        precisions.append(targets[top_k, tag].mean())
    return torch.stack(precisions)


def similar_search_precision(embeddings: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """Use every image as a query: find its k most similar other images.

    A result counts as correct if it shares at least one tag with the query.
    Returns the average share of correct results.
    """
    similarity = embeddings @ embeddings.T             # (images, images)
    similarity.fill_diagonal_(-float("inf"))            # an image shouldn't find itself
    top_k = similarity.topk(k, dim=1).indices           # (images, k)

    shared_tags = targets[top_k] * targets.unsqueeze(1) # (images, k, 47): 1 where both have the tag
    correct = shared_tags.sum(dim=2) > 0                # (images, k)
    return correct.float().mean().item()


def save_checkpoint(model: nn.Module, arch: str, classes: List[str], path: Path):
    """Same format as the single-label models, plus multilabel=True."""
    torch.save({"arch": arch, "state_dict": model.state_dict(), "classes": classes, "multilabel": True}, path)
