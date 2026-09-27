"""Data loading assembly for FER-2013.

Owns: transforms, class weights, sampler, and DataLoader construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T

from fer_2013.data.dataset import FER2013Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESIZE_TO = (224, 224)
N_CLASSES = 7


class _ReplicateTo3Ch:
    """Picklable replacement for T.Lambda (works with num_workers > 0)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(3, 1, 1) if x.shape[0] == 1 else x


def build_transforms(split: str) -> T.Compose:
    """Return the transform pipeline for a split.

    Order: resize -> replicate to 3ch -> (train-only aug) -> normalize.
    Train gets augmentation; val and test do not.
    """
    pre: list[object] = [
        T.Resize(
            RESIZE_TO,
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        _ReplicateTo3Ch(),
    ]
    post: list[object] = [
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if split == "train":
        aug: list[object] = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return T.Compose(pre + aug + post)

    return T.Compose(pre + post)


def build_class_weights(y: np.ndarray, n_classes: int = N_CLASSES) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss.

    weight_c = N / (C * count_c)
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weights = len(y) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_dataloader(
    processed_dir: Path,
    split: str,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    balanced: bool | None = None,
    shuffle: bool | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a DataLoader for one split.

    Defaults:
        - balanced sampling on train, off elsewhere.
        - shuffle on train only when balanced=False.
    """
    if balanced is None:
        balanced = split == "train"
    if shuffle is None:
        shuffle = split == "train" and not balanced

    ds = FER2013Dataset(
        processed_dir,
        split=split,
        transform=build_transforms(split),
    )

    sampler: WeightedRandomSampler | None = None
    if balanced:
        y = np.load(processed_dir / f"y_{split}.npy")
        counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
        counts = np.where(counts == 0, 1.0, counts)
        sample_weights = 1.0 / counts[y]
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(ds),
            replacement=True,
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "N_CLASSES",
    "RESIZE_TO",
    "build_class_weights",
    "build_transforms",
    "make_dataloader",
]
