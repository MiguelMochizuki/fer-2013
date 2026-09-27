#!/usr/bin/env python3
"""CLI to download FER-2013 (CSV format) via kagglehub.

Usage:
    uv run python scripts/download_data.py --csv-path data/raw/
    uv run python scripts/download_data.py --csv-path data/raw/ --force
    uv run python scripts/download_data.py --csv-path data/raw/ -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from fer_2013.data.download import (
    DatasetLayoutError,
    MissingCredentialsError,
    download_fer2013,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FER-2013 (CSV) via kagglehub.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Directory where fer2013.csv will be saved (e.g. data/raw/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the CSV already exists.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log at DEBUG level.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("download_data")

    try:
        csv_path = download_fer2013(args.csv_path, force=args.force)
    except MissingCredentialsError as exc:
        log.error("Missing credentials: %s", exc)
        return 2
    except DatasetLayoutError as exc:
        log.error("Unexpected dataset layout: %s", exc)
        return 3

    log.info("CSV available at: %s", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
