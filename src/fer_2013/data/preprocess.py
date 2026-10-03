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


# FER+ vote columns in our label order; the other three (contempt, unknown, NF) only
# decide whether a row is kept.
FERPLUS_COLUMNS: Final[tuple[str, ...]] = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
)
_FERPLUS_DROPPED: Final[tuple[str, ...]] = ("contempt", "unknown", "NF")


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


def preprocess_ferplus(
    csv_path: Path,
    ferplus_csv_path: Path,
    out_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Write FER-2013 arrays relabelled with the FER+ crowd votes.

    The FER+ rows are aligned one to one with fer2013.csv. Rows whose majority vote
    is contempt, unknown or not-a-face are dropped. Besides the usual
    `X_<split>.npy` and `y_<split>.npy` (the argmax of the votes), each split gets
    `y_<split>_soft.npy`: (N, 7) float32 vote fractions in `EMOTION_LABELS` order.

    Args:
        csv_path: path to fer2013.csv.
        ferplus_csv_path: path to fer2013new.csv from github.com/microsoft/FERPlus.
        out_dir: directory where the .npy files will be written.
        force: if True, overwrite existing .npy files.

    Returns:
        Mapping of output filename to the written path.

    Raises:
        FileNotFoundError: one of the CSVs does not exist.
        PreprocessError: the CSVs are not aligned or lack columns.
    """
    ferplus_csv_path = Path(ferplus_csv_path)
    if not ferplus_csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {ferplus_csv_path}")
    out_dir = Path(out_dir).expanduser().resolve()
    names = [f"{kind}_{s}.npy" for s in SPLIT_MAP.values() for kind in ("X", "y")]
    names += [f"y_{s}_soft.npy" for s in SPLIT_MAP.values()]
    expected = [out_dir / n for n in names]
    if all(p.exists() for p in expected) and not force:
        return {p.name: p for p in expected}

    fer = pd.read_csv(csv_path)
    plus = pd.read_csv(ferplus_csv_path)
    missing = {"Usage", *FERPLUS_COLUMNS, *_FERPLUS_DROPPED} - set(plus.columns)
    if missing:
        raise PreprocessError(f"FER+ CSV is missing columns: {sorted(missing)}")
    if len(fer) != len(plus) or not (fer["Usage"] == plus["Usage"]).all():
        raise PreprocessError("fer2013.csv and the FER+ CSV are not row-aligned.")

    votes = plus[[*FERPLUS_COLUMNS, *_FERPLUS_DROPPED]].to_numpy(dtype=np.float64)
    keep = votes.argmax(axis=1) < len(FERPLUS_COLUMNS)
    soft = votes[:, : len(FERPLUS_COLUMNS)]
    soft = (soft / soft.sum(axis=1, keepdims=True).clip(min=1)).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    for usage, split in SPLIT_MAP.items():
        rows = (fer["Usage"] == usage).to_numpy() & keep
        X = _parse_pixels_column(fer.loc[rows, "pixels"])
        np.save(out_dir / f"X_{split}.npy", X)
        np.save(out_dir / f"y_{split}.npy", soft[rows].argmax(axis=1).astype(np.int64))
        np.save(out_dir / f"y_{split}_soft.npy", soft[rows])
    return {p.name: p for p in expected}


__all__ = [
    "EMOTION_LABELS",
    "FERPLUS_COLUMNS",
    "IMAGE_SHAPE",
    "PIXEL_COUNT",
    "SPLIT_MAP",
    "MalformedRowError",
    "PreprocessError",
    "preprocess_fer2013",
    "preprocess_ferplus",
]
