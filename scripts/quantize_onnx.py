#!/usr/bin/env python3
"""CLI to quantize the exported fp32 classifier to int8 and check it.

Usage:
    uv run python scripts/quantize_onnx.py \\
        --model models/fer_resnet18_fp32.onnx --out models/fer_resnet18.onnx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from fer_2013.models.quantize import (
    QuantizationError,
    quantize_classifier,
    verify_quantized,
)

logger = logging.getLogger("quantize_onnx")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize the classifier to int8.")
    parser.add_argument(
        "--model", type=Path, required=True, help="fp32 model from export_onnx.py"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="where to write the int8 model"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--calibration-images", type=int, default=300)
    parser.add_argument(
        "--verify-images",
        type=int,
        default=1000,
        help="validation faces for the quality gate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    train = np.load(args.data_dir / "X_train.npy")
    quantize_classifier(
        args.model, args.out, train, n_calibration=args.calibration_images
    )
    n = args.verify_images
    x_val = np.load(args.data_dir / "X_val.npy")[:n]
    y_val = np.load(args.data_dir / "y_val.npy")[:n]
    try:
        report = verify_quantized(args.model, args.out, x_val, y_val)
    except QuantizationError as error:
        args.out.unlink(missing_ok=True)
        logger.error("%s", error)
        return 1
    logger.info("%s -> %s: %s", args.model, args.out, json.dumps(report))
    logger.info(
        "size %.1f MB -> %.1f MB",
        args.model.stat().st_size / 1e6,
        args.out.stat().st_size / 1e6,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
