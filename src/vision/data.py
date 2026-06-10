"""Dataset loading and transforms for Oxford Flowers102.

Flowers102 splits:
  train  1,020 images (10 per class × 102 classes)
  val    1,020 images
  test   6,149 images
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102

ROOT = Path(__file__).parents[2]
DATA_DIR = ROOT / "data"

NUM_CLASSES = 102
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = Flowers102(
        root=DATA_DIR, split="train", transform=get_transforms(True), download=download
    )
    val_ds = Flowers102(
        root=DATA_DIR, split="val", transform=get_transforms(False), download=download
    )
    test_ds = Flowers102(
        root=DATA_DIR, split="test", transform=get_transforms(False), download=download
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
