#!/usr/bin/env python3
"""CLI to evaluate a trained checkpoint on a split.

Usage:
    uv run python scripts/evaluate.py \\
        --checkpoint checkpoints/best.pt \\
        --processed-dir data/processed \\
        --split test \\
        --reports-dir reports
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from fer_2013.data.datamodule import make_dataloader
from fer_2013.evaluation.evaluate import (
    EMOTION_LABELS,
    compute_metrics,
    load_checkpoint,
    per_class_report,
    predict,
)
from fer_2013.models.cnn import build_resnet18


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a FER-2013 checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--no-save-arrays",
        action="store_true",
        help="Skip saving preds/targets/probs .npy files.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("evaluate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    model = build_resnet18(num_classes=len(EMOTION_LABELS), pretrained=False)
    ckpt = load_checkpoint(args.checkpoint, model, map_location=device)
    log.info(
        "loaded %s (epoch=%s, best_val_score=%s)",
        args.checkpoint,
        ckpt.get("epoch"),
        ckpt.get("best_val_score"),
    )
    model.to(device)

    loader = make_dataloader(
        args.processed_dir,
        args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    log.info("running inference on split=%s", args.split)
    preds = predict(model, loader, device)
    metrics = compute_metrics(preds.preds, preds.targets, n_classes=len(EMOTION_LABELS))

    print()
    print(per_class_report(metrics, labels=EMOTION_LABELS))
    print()

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.reports_dir / f"{args.split}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "split": args.split,
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "labels": list(EMOTION_LABELS),
                "per_class_precision": metrics.per_class_precision,
                "per_class_recall": metrics.per_class_recall,
                "per_class_f1": metrics.per_class_f1,
                "per_class_support": metrics.per_class_support,
            },
            f,
            indent=2,
        )
    log.info("wrote %s", metrics_path)

    assert metrics.confusion is not None
    np.save(args.reports_dir / f"{args.split}_confusion.npy", metrics.confusion)

    if not args.no_save_arrays:
        np.save(args.reports_dir / f"{args.split}_preds.npy", preds.preds)
        np.save(args.reports_dir / f"{args.split}_targets.npy", preds.targets)
        np.save(args.reports_dir / f"{args.split}_probs.npy", preds.probs)
        log.info("saved prediction arrays to %s", args.reports_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
