"""Preprocess FER-2013 CSV into .npy arrays for fast loading.

Input:  data/raw/fer2013.csv  (columns: emotion, pixels, Usage)
Output: <out_dir>/X_train.npy, y_train.npy, X_val.npy, y_val.npy,
        X_test.npy, y_test.npy
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

IMAGE_SHAPE: Final[tuple[int, int]] = (48, 48)
PIXEL_COUNT: Final[int] = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]  # 2304

# Maps the CSV "Usage" column to our split names.
SPLIT_MAP: Final[dict[str, str]] = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test",
}

# Standard FER-2013 label ordering.
EMOTION_LABELS: Final[tuple[str, ...]] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)


class PreprocessError(RuntimeError):
    """Base error for preprocessing."""


class MalformedRowError(PreprocessError):
    """A row has a pixel string that does not parse to 48*48 uint8."""


def _parse_pixels(raw: str) -> np.ndarray:
    """Parse a space-separated pixel string into a (48, 48) uint8 array."""
    values = np.array(raw.split(), dtype=np.uint8)
    if values.size != PIXEL_COUNT:
        raise MalformedRowError(f"Expected {PIXEL_COUNT} pixels, got {values.size}.")
    return values.reshape(IMAGE_SHAPE)


def _parse_pixels_column(series: pd.Series) -> np.ndarray:
    """Vectorize _parse_pixels over a pandas Series."""
    arrays = [_parse_pixels(s) for s in series]
    return np.stack(arrays, axis=0)


def preprocess_fer2013(
    csv_path: Path,
    out_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Read the FER-2013 CSV and write one .npy per (split, array).

    Args:
        csv_path: path to fer2013.csv.
        out_dir: directory where the .npy files will be written.
        force: if True, overwrite existing .npy files.

    Returns:
        Mapping of output filename to the written path.

    Raises:
        FileNotFoundError: csv_path does not exist.
        PreprocessError: malformed pixels or unknown Usage values.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    expected = [
        out_dir / name
        for name in (
            "X_train.npy",
            "y_train.npy",
            "X_val.npy",
            "y_val.npy",
            "X_test.npy",
            "y_test.npy",
        )
    ]
    if all(p.exists() for p in expected) and not force:
        return {p.name: p for p in expected}

    df = pd.read_csv(csv_path)
    missing = {"emotion", "pixels", "Usage"} - set(df.columns)
    if missing:
        raise PreprocessError(f"CSV is missing columns: {sorted(missing)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    unknown = set(df["Usage"].unique()) - set(SPLIT_MAP)
    if unknown:
        raise PreprocessError(f"Unknown Usage values: {sorted(unknown)}")

    for usage, split in SPLIT_MAP.items():
        subset = df[df["Usage"] == usage]

        if len(subset) == 0:
            X = np.empty((0, *IMAGE_SHAPE), dtype=np.uint8)
            y = np.empty((0,), dtype=np.int64)
        else:
            X = _parse_pixels_column(subset["pixels"])
            y = subset["emotion"].to_numpy(dtype=np.int64)

        x_path = out_dir / f"X_{split}.npy"
        y_path = out_dir / f"y_{split}.npy"
        np.save(x_path, X)
        np.save(y_path, y)
        written[x_path.name] = x_path
        written[y_path.name] = y_path

    return written


__all__ = [
    "EMOTION_LABELS",
    "IMAGE_SHAPE",
    "PIXEL_COUNT",
    "SPLIT_MAP",
    "MalformedRowError",
    "PreprocessError",
    "preprocess_fer2013",
]
