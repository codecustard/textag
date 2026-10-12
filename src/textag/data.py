"""Data: the Describable Textures Dataset (DTD) from torchvision.

47 texture classes (banded, cracked, marbled, scaly, woven, ...), 5,640 images.
torchvision ships 10 official partitions; partition 1 gives train / val / test of
1,880 images each (40 per class).
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMG_SIZE = 224

# Training: random crops/flips/rotations so the model sees new variations each epoch.
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),          # textures have no "up"
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Validation / test / app: deterministic, no augmentation.
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def dataset(split: str, partition: int = 1):
    tf = train_tf if split == "train" else eval_tf
    return datasets.DTD(DATA_DIR, split=split, partition=partition, transform=tf, download=True)


def loaders(batch_size: int = 32, num_workers: int = 2, partition: int = 1):
    """train (shuffled) / val / test DataLoaders."""
    ds = {s: dataset(s, partition) for s in ("train", "val", "test")}
    return {
        s: DataLoader(d, batch_size=batch_size, shuffle=(s == "train"), num_workers=num_workers)
        for s, d in ds.items()
    }


def class_names(partition: int = 1) -> list[str]:
    return datasets.DTD(DATA_DIR, split="test", partition=partition, download=True).classes
