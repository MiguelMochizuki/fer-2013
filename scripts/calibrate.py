#!/usr/bin/env python3
"""CLI to calibrate a trained checkpoint with temperature scaling.

Fits one temperature on a held-out split (validation by default), reports the
calibration metrics before and after on the fit and evaluation splits, and
writes `calibration.json` plus a reliability diagram. The JSON feeds
`scripts/export_onnx.py --calibration`.

Usage:
    uv run python scripts/calibrate.py --checkpoint checkpoints/ferplus/best.pt
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
from fer_2013.evaluation.calibration import (
    brier_score,
    calibrated_probs,
    collect_logits,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    reliability_bins,
)
from fer_2013.evaluation.evaluate import EMOTION_LABELS, load_checkpoint
from fer_2013.evaluation.plots import plot_reliability_diagram
from fer_2013.models.cnn import build_resnet18

logger = logging.getLogger("calibrate")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temperature-scale a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--fit-split", choices=["val"], default="val")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args(argv)


def _metrics(
    logits: np.ndarray, targets: np.ndarray, temperature: float
) -> dict[str, float]:
    probs = calibrated_probs(logits, temperature)
    return {
        "accuracy": float((probs.argmax(axis=1) == targets).mean()),
        "ece": expected_calibration_error(probs, targets),
        "nll": negative_log_likelihood(logits, targets, temperature),
        "brier": brier_score(probs, targets),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the temperature-scaling calibration command line tool.

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=len(EMOTION_LABELS), pretrained=False)
    load_checkpoint(args.checkpoint, model, map_location=device)

    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in dict.fromkeys([args.fit_split, args.eval_split]):
        loader = make_dataloader(
            args.processed_dir,
            split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        splits[split] = collect_logits(model, loader, device)

    fit_logits, fit_targets = splits[args.fit_split]
    temperature = fit_temperature(fit_logits, fit_targets)
    logger.info(
        "fitted T = %.3f on %s (n=%d)", temperature, args.fit_split, len(fit_targets)
    )

    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "temperature": temperature,
        "fit_split": args.fit_split,
        "n_fit": int(len(fit_targets)),
        "metrics": {},
    }
    print(
        f"{'split':<6} {'':<11} {'acc':>6} {'ECE':>6} {'NLL':>6} {'Brier':>6} {'conf':>6}"
    )
    for split, (logits, targets) in splits.items():
        raw, cal = (
            _metrics(logits, targets, 1.0),
            _metrics(logits, targets, temperature),
        )
        report["metrics"][split] = {"raw": raw, "calibrated": cal}  # type: ignore[index]
        for name, m in (("raw", raw), ("calibrated", cal)):
            print(
                f"{split:<6} {name:<11} {m['accuracy']:>6.3f} {m['ece']:>6.3f} "
                f"{m['nll']:>6.3f} {m['brier']:>6.3f} {m['mean_confidence']:>6.3f}"
            )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.reports_dir / "calibration.json"
    json_path.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", json_path)

    eval_logits, eval_targets = splits[args.eval_split]
    plot_path = plot_reliability_diagram(
        {
            "raw": reliability_bins(calibrated_probs(eval_logits, 1.0), eval_targets),
            f"temperature scaled (T={temperature:.2f})": reliability_bins(
                calibrated_probs(eval_logits, temperature), eval_targets
            ),
        },
        args.reports_dir / f"{args.eval_split}_reliability.png",
    )
    logger.info("wrote %s", plot_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
