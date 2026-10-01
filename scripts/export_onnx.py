#!/usr/bin/env python3
"""CLI to export a trained checkpoint to ONNX for the serving image.

Usage:
    uv run python scripts/export_onnx.py \\
        --checkpoint checkpoints/best.pt --out-dir models/
"""

from __future__ import annotations

import argparse
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.checkpoint.exists():
        logger.error("checkpoint not found: %s", args.checkpoint)
        return 1

    model = build_resnet18(num_classes=len(EMOTION_LABELS), pretrained=False)
    load_checkpoint(args.checkpoint, model)
    onnx_path, fc_path = export_classifier(model, args.out_dir)
    logger.info("wrote %s and %s (parity with PyTorch verified)", onnx_path, fc_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
