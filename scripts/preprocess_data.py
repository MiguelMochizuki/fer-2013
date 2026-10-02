#!/usr/bin/env python3
"""CLI to preprocess FER-2013 CSV into .npy arrays.

Usage:
    uv run python scripts/preprocess_data.py \\
        --csv-path data/raw/fer2013.csv \\
        --out-dir data/processed/
    uv run python scripts/preprocess_data.py \\
        --csv-path data/raw/fer2013.csv \\
        --out-dir data/processed/ --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fer_2013.data.preprocess import (
    MalformedRowError,
    PreprocessError,
    preprocess_fer2013,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess FER-2013 CSV into .npy arrays.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Path to fer2013.csv (e.g. data/raw/fer2013.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where .npy files will be written (e.g. data/processed/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .npy files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log at DEBUG level.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the CSV preprocessing command line tool.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("preprocess_data")

    try:
        written = preprocess_fer2013(args.csv_path, args.out_dir, force=args.force)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except MalformedRowError as exc:
        log.error("Malformed row in CSV: %s", exc)
        return 3
    except PreprocessError as exc:
        log.error("Preprocessing failed: %s", exc)
        return 4

    for path in sorted(written.values()):
        log.info("wrote %s", path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
