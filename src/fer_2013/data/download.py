"""Download FER-2013 (CSV format) via kagglehub.

Dataset: https://www.kaggle.com/datasets/deadskull7/fer2013
Single file: fer2013.csv (columns: emotion, pixels, Usage)

Authentication: environment variable KAGGLE_KEY (legacy key) or
KAGGLE_API_TOKEN (new scoped token).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Final

import kagglehub

DATASET_ID: Final[str] = "deadskull7/fer2013"
CSV_FILENAME: Final[str] = "fer2013.csv"


class DownloadError(RuntimeError):
    """Base error for FER-2013 download."""


class MissingCredentialsError(DownloadError):
    """KAGGLE_KEY / KAGGLE_API_TOKEN not configured."""


class DatasetLayoutError(DownloadError):
    """kagglehub returned a folder without the expected CSV."""


def _check_credentials() -> None:
    """Raise MissingCredentialsError if no credentials are in the env."""
    has_legacy = bool(os.environ.get("KAGGLE_KEY"))
    has_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
    if not (has_legacy or has_token):
        raise MissingCredentialsError(
            "Missing Kaggle credentials. Set KAGGLE_KEY (legacy) "
            "or KAGGLE_API_TOKEN in the environment or in a .env file."
        )


def _kaggle_download(dataset_id: str) -> Path:
    """Thin wrapper around kagglehub (isolated so tests can mock it)."""
    return Path(kagglehub.dataset_download(dataset_id))


def download_fer2013(dest_dir: Path, *, force: bool = False) -> Path:
    """Download fer2013.csv into dest_dir/fer2013.csv.

    Args:
        dest_dir: folder where the CSV will be installed.
        force: if True, re-download even if the CSV already exists.

    Returns:
        Full path of the installed CSV.

    Raises:
        MissingCredentialsError: no KAGGLE_KEY nor KAGGLE_API_TOKEN.
        DatasetLayoutError: kagglehub did not return fer2013.csv.
    """
    _check_credentials()

    dest_dir = Path(dest_dir).expanduser().resolve()
    dest_csv = dest_dir / CSV_FILENAME

    if dest_csv.exists() and not force:
        return dest_csv

    if dest_csv.exists() and force:
        dest_csv.unlink()

    dest_dir.mkdir(parents=True, exist_ok=True)

    kaggle_dir = _kaggle_download(DATASET_ID)
    src_csv = kaggle_dir / CSV_FILENAME

    if not src_csv.exists():
        # Some mirrors nest the CSV one level deeper; try one rglob.
        candidates = list(kaggle_dir.rglob(CSV_FILENAME))
        if not candidates:
            raise DatasetLayoutError(
                f"{CSV_FILENAME} not found in {kaggle_dir}. "
                f"Contents: {list(kaggle_dir.iterdir())!r}"
            )
        src_csv = candidates[0]

    shutil.copy2(src_csv, dest_csv)
    return dest_csv


__all__ = [
    "CSV_FILENAME",
    "DATASET_ID",
    "DatasetLayoutError",
    "DownloadError",
    "MissingCredentialsError",
    "download_fer2013",
]
