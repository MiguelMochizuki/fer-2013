"""Tests for fer_2013.data.dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from fer_2013.data.dataset import FER2013Dataset


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    """Build a tiny processed/ dir with train/val/test arrays."""
    d = tmp_path / "processed"
    d.mkdir()
    for split, n in (("train", 3), ("val", 2), ("test", 1)):
        X = np.zeros((n, 48, 48), dtype=np.uint8)
        y = np.arange(n, dtype=np.int64)
        np.save(d / f"X_{split}.npy", X)
        np.save(d / f"y_{split}.npy", y)
    return d


def test_len_matches_number_of_samples(processed_dir: Path) -> None:
    ds = FER2013Dataset(processed_dir, split="train")
    assert len(ds) == 3


def test_is_torch_dataset(processed_dir: Path) -> None:
    ds = FER2013Dataset(processed_dir, split="train")
    assert isinstance(ds, Dataset)


def test_getitem_returns_image_and_label(processed_dir: Path) -> None:
    ds = FER2013Dataset(processed_dir, split="train")
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 48, 48)
    assert isinstance(label, torch.Tensor)
    assert label.item() == 0


def test_getitem_returns_float_tensor_normalized(processed_dir: Path) -> None:
    ds = FER2013Dataset(processed_dir, split="train")
    image, _ = ds[0]
    assert image.dtype == torch.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_label_dtype_is_long(processed_dir: Path) -> None:
    ds = FER2013Dataset(processed_dir, split="train")
    _, label = ds[0]
    assert label.dtype == torch.long


def test_missing_split_raises_clear_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        FER2013Dataset(empty, split="train")


def test_mismatched_x_y_length_raises(tmp_path: Path) -> None:
    d = tmp_path / "processed"
    d.mkdir()
    np.save(d / "X_train.npy", np.zeros((3, 48, 48), dtype=np.uint8))
    np.save(d / "y_train.npy", np.zeros((2,), dtype=np.int64))
    with pytest.raises(ValueError):
        FER2013Dataset(d, split="train")


def test_transform_is_applied(processed_dir: Path) -> None:
    calls: list[torch.Tensor] = []

    def transform(x: torch.Tensor) -> torch.Tensor:
        calls.append(x)
        return x

    ds = FER2013Dataset(processed_dir, split="train", transform=transform)
    ds[0]
    assert len(calls) == 1
