"""Tests for fer_2013.data.datamodule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms as T

from fer_2013.data.datamodule import (
    build_class_weights,
    build_transforms,
    make_dataloader,
)


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    """Tiny processed/ dir with all three splits."""
    d = tmp_path / "processed"
    d.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("train", 8), ("val", 4), ("test", 4)):
        X = rng.integers(0, 256, size=(n, 48, 48), dtype=np.uint8)
        y = rng.integers(0, 7, size=(n,), dtype=np.int64)
        np.save(d / f"X_{split}.npy", X)
        np.save(d / f"y_{split}.npy", y)
    return d


# ==============================
# build_transforms
# ==============================


def test_train_transform_output_shape_and_dtype() -> None:
    x = torch.zeros(1, 48, 48, dtype=torch.float32)
    out = build_transforms("train")(x)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_val_transform_is_deterministic() -> None:
    x = torch.full((1, 48, 48), 100 / 255.0, dtype=torch.float32)
    t = build_transforms("val")
    a = t(x)
    b = t(x)
    assert torch.equal(a, b)


def test_train_transform_is_stochastic() -> None:
    """Two calls on the same input should differ because of augmentation."""
    torch.manual_seed(0)
    x = torch.full((1, 48, 48), 100 / 255.0, dtype=torch.float32)
    t = build_transforms("train")
    a = t(x)
    b = t(x)
    assert not torch.equal(a, b)


def test_train_pipeline_uses_bicubic_resize() -> None:
    """Confirm bicubic interpolation is configured."""
    t = build_transforms("train")
    resize = next(op for op in t.transforms if isinstance(op, T.Resize))
    assert resize.interpolation == T.InterpolationMode.BICUBIC


def test_val_pipeline_uses_bicubic_resize() -> None:
    t = build_transforms("val")
    resize = next(op for op in t.transforms if isinstance(op, T.Resize))
    assert resize.interpolation == T.InterpolationMode.BICUBIC


def test_val_transform_has_no_random_ops() -> None:
    """Val pipeline must not contain augmentation operators."""
    t = build_transforms("val")
    names = {type(op).__name__ for op in t.transforms}
    forbidden = {
        "RandomHorizontalFlip",
        "RandomAffine",
        "ColorJitter",
        "RandomRotation",
        "RandomErasing",
    }
    assert forbidden.isdisjoint(names)


# ==============================
# build_class_weights
# ==============================


def test_class_weights_inverse_frequency() -> None:
    y = np.array([0] * 100 + [1] * 50 + [2] * 25, dtype=np.int64)
    w = build_class_weights(y, n_classes=3)
    # weight_0 = 175 / (3 * 100), weight_1 = 175 / (3 * 50), weight_2 = 175 / (3 * 25)
    assert w.shape == (3,)
    assert w[0] < w[1] < w[2]
    assert torch.allclose(
        w * torch.tensor([100.0, 50.0, 25.0]),
        torch.full((3,), 175.0 / 3.0),
        rtol=1e-5,
    )


def test_class_weights_handles_missing_class() -> None:
    """A class with zero samples must not cause division by zero."""
    y = np.array([0] * 10 + [2] * 5, dtype=np.int64)
    w = build_class_weights(y, n_classes=3)
    assert w.shape == (3,)
    assert torch.isfinite(w).all()


# ==============================
# make_dataloader
# ==============================


def test_dataloader_batch_contract(processed_dir: Path) -> None:
    loader = make_dataloader(processed_dir, "train", batch_size=4, num_workers=0)
    x, y = next(iter(loader))
    assert x.shape == (4, 3, 224, 224)
    assert x.dtype == torch.float32
    assert y.shape == (4,)
    assert y.dtype == torch.int64


def test_dataloader_val_is_not_shuffled(processed_dir: Path) -> None:
    loader = make_dataloader(processed_dir, "val", batch_size=4, num_workers=0)
    # Run twice; order should be identical for val.
    order_a = [y.tolist() for _, y in loader]
    order_b = [y.tolist() for _, y in loader]
    assert order_a == order_b


def test_dataloader_train_is_balanced(processed_dir: Path) -> None:
    """With balanced=True, the sampler should produce varied batches."""
    loader = make_dataloader(
        processed_dir,
        "train",
        batch_size=8,
        num_workers=0,
        balanced=True,
    )
    assert isinstance(loader.sampler, WeightedRandomSampler)


def test_dataloader_val_has_no_sampler(processed_dir: Path) -> None:
    loader = make_dataloader(processed_dir, "val", batch_size=4, num_workers=0)
    assert not isinstance(loader.sampler, WeightedRandomSampler)


def test_dataloader_works_with_multiple_workers(processed_dir: Path) -> None:
    """Regression: transform pipeline must be picklable for num_workers > 0."""
    loader = make_dataloader(processed_dir, "train", batch_size=4, num_workers=2)
    x, y = next(iter(loader))
    assert x.shape == (4, 3, 224, 224)
    assert y.shape == (4,)
