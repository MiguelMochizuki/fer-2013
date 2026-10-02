#!/usr/bin/env python3
"""CLI to export a trained checkpoint to ONNX for the serving image.

Usage:
    uv run python scripts/export_onnx.py \\
        --checkpoint checkpoints/best.pt --out-dir models/ \\
        --calibration reports/calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from fer_2013.evaluation.evaluate import EMOTION_LABELS, load_checkpoint
from fer_2013.models.cnn import build_resnet18
from fer_2013.models.export import export_classifier

logger = logging.getLogger("export_onnx")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the classifier to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="calibration.json from scripts/calibrate.py; without it the probs "
        "output is a plain softmax (temperature 1).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the ONNX export command line tool.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.checkpoint.exists():
        logger.error("checkpoint not found: %s", args.checkpoint)
        return 1

    temperature = 1.0
    if args.calibration is not None:
        temperature = float(json.loads(args.calibration.read_text())["temperature"])
    else:
        logger.warning("no --calibration given: exporting with temperature 1.0")

    model = build_resnet18(num_classes=len(EMOTION_LABELS), pretrained=False)
    load_checkpoint(args.checkpoint, model)
    onnx_path, fc_path = export_classifier(model, args.out_dir, temperature)
    logger.info(
        "wrote %s and %s (temperature %.3f, parity with PyTorch verified)",
        onnx_path,
        fc_path,
        temperature,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
