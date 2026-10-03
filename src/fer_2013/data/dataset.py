"""PyTorch Dataset for the preprocessed FER-2013 arrays.

Reads the .npy files produced by fer_2013.data.preprocess:
    <processed_dir>/X_<split>.npy  uint8   (N, 48, 48)
    <processed_dir>/y_<split>.npy  int64   (N,)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

Transform = Callable[[torch.Tensor], torch.Tensor]

SPLITS = ("train", "val", "test")
IMAGE_SHAPE = (48, 48)


class FER2013Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset over one split of the preprocessed FER-2013 arrays.

    Args:
        processed_dir: directory containing X_<split>.npy and y_<split>.npy.
        split: one of "train", "val", "test".
        transform: optional callable applied to the image tensor.
        soft: return the (7,) float vote fractions from `y_<split>_soft.npy`
            (written by `preprocess_ferplus`) instead of the class index.

    Raises:
        FileNotFoundError: the split files are missing.
        ValueError: X and y have different lengths.
    """

    def __init__(
        self,
        processed_dir: Path,
        *,
        split: str = "train",
        transform: Transform | None = None,
        soft: bool = False,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split!r}. Expected one of {SPLITS}.")

        self.processed_dir = Path(processed_dir)
        self.split = split
        self.transform = transform

        self._X = self._load("X", split)
        self._y = self._load("y_soft" if soft else "y", split)
        self._soft = soft

        if len(self._X) != len(self._y):
            raise ValueError(
                f"X and y length mismatch for split {split!r}: "
                f"{len(self._X)} vs {len(self._y)}"
            )

    def _load(self, kind: str, split: str) -> np.ndarray:
        path = self.processed_dir / (
            f"y_{split}_soft.npy" if kind == "y_soft" else f"{kind}_{split}.npy"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing array: {path}")
        arr: np.ndarray = np.load(path)
        return arr

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._X[idx]  # (48, 48) uint8
        y = self._y[idx]

        x_t = torch.from_numpy(x).float().unsqueeze(0) / 255.0
        y_t = torch.tensor(y, dtype=torch.float32 if self._soft else torch.long)

        if self.transform is not None:
            x_t = self.transform(x_t)

        return x_t, y_t


__all__ = ["SPLITS", "FER2013Dataset", "Transform"]
