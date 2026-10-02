#!/usr/bin/env python3
"""CLI to generate Grad-CAM visualizations for a trained checkpoint.

Produces two grids: the most confident correct predictions, and the most
confident misclassifications, each with a Grad-CAM overlay on the logit of
the predicted class.

Usage:
    uv run python scripts/gradcam_report.py \\
        --checkpoint checkpoints/best.pt \\
        --split test \\
        --reports-dir reports
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch

from fer_2013.data.datamodule import build_transforms, make_dataloader
from fer_2013.data.dataset import FER2013Dataset
from fer_2013.evaluation.evaluate import EMOTION_LABELS, load_checkpoint, predict
from fer_2013.evaluation.gradcam import (
    CAMMethod,
    denormalize,
    generate_heatmaps,
    overlay,
)
from fer_2013.evaluation.plots import plot_image_grid
from fer_2013.models.cnn import build_resnet18


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM report grids.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--n-correct", type=int, default=8)
    parser.add_argument("--n-misclassified", type=int, default=8)
    parser.add_argument(
        "--method", choices=["gradcam", "gradcam++"], default="gradcam++"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Grad-CAM++ report command line tool.

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
    log = logging.getLogger("gradcam_report")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    model = build_resnet18(num_classes=len(EMOTION_LABELS), pretrained=False)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.to(device)

    loader = make_dataloader(
        args.processed_dir,
        args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    log.info("running inference on split=%s", args.split)
    preds = predict(model, loader, device)

    confidence = preds.probs[np.arange(len(preds.preds)), preds.preds]
    correct_mask = preds.preds == preds.targets

    correct_idx = np.where(correct_mask)[0]
    correct_idx = correct_idx[np.argsort(-confidence[correct_idx])][: args.n_correct]

    wrong_idx = np.where(~correct_mask)[0]
    wrong_idx = wrong_idx[np.argsort(-confidence[wrong_idx])][: args.n_misclassified]

    dataset = FER2013Dataset(
        args.processed_dir, split=args.split, transform=build_transforms(args.split)
    )

    def render(indices: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
        xs = torch.stack([dataset[int(i)][0] for i in indices]).to(device)
        class_indices = [int(preds.preds[i]) for i in indices]
        heatmaps = generate_heatmaps(
            model, xs, class_indices, method=cast(CAMMethod, args.method)
        )

        images = [overlay(denormalize(xs[j]), heatmaps[j]) for j in range(len(indices))]
        titles = [
            f"pred={EMOTION_LABELS[preds.preds[i]]} "
            f"true={EMOTION_LABELS[preds.targets[i]]} "
            f"({confidence[i]:.2f})"
            for i in indices
        ]
        return images, titles

    args.reports_dir.mkdir(parents=True, exist_ok=True)

    correct_images, correct_titles = render(correct_idx)
    correct_path = plot_image_grid(
        correct_images,
        correct_titles,
        args.reports_dir / f"{args.split}_gradcam_correct.png",
    )
    log.info("wrote %s", correct_path)

    wrong_images, wrong_titles = render(wrong_idx)
    wrong_path = plot_image_grid(
        wrong_images,
        wrong_titles,
        args.reports_dir / f"{args.split}_gradcam_misclassified.png",
    )
    log.info("wrote %s", wrong_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
