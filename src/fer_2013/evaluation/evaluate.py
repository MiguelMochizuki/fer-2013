"""Evaluation primitives: checkpoint loading, inference, metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

EMOTION_LABELS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)


@dataclass
class Predictions:
    """Outputs of one inference pass over a loader."""

    preds: np.ndarray  # (N,) int64
    targets: np.ndarray  # (N,) int64
    probs: np.ndarray  # (N, C) float32, softmax


@dataclass
class Metrics:
    """Aggregated metrics for one split."""

    accuracy: float
    macro_f1: float
    per_class_precision: list[float] = field(default_factory=list)
    per_class_recall: list[float] = field(default_factory=list)
    per_class_f1: list[float] = field(default_factory=list)
    per_class_support: list[int] = field(default_factory=list)
    confusion: np.ndarray | None = None


def load_checkpoint(
    path: Path, model: nn.Module, map_location: str | torch.device = "cpu"
) -> dict[str, object]:
    """Load a checkpoint produced by training.save_checkpoint.

    Returns the checkpoint dict (metadata included).
    """
    ckpt: dict[str, object] = torch.load(
        path, map_location=map_location, weights_only=False
    )
    state_dict = cast(dict[str, torch.Tensor], ckpt["model"])
    model.load_state_dict(state_dict)
    return ckpt


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Predictions:
    """Run inference over a loader. Returns preds, targets, softmax probs."""
    model.eval()
    model.to(device)

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return Predictions(
        preds=np.concatenate(all_preds),
        targets=np.concatenate(all_targets),
        probs=np.concatenate(all_probs).astype(np.float32),
    )


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    n_classes: int = 7,
) -> Metrics:
    """Accuracy, macro-F1, per-class P/R/F1, and confusion matrix."""
    labels = list(range(n_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        labels=labels,
        zero_division=0,
    )
    return Metrics(
        accuracy=float(accuracy_score(targets, preds)),
        macro_f1=float(
            f1_score(targets, preds, average="macro", labels=labels, zero_division=0)
        ),
        per_class_precision=[float(x) for x in precision],
        per_class_recall=[float(x) for x in recall],
        per_class_f1=[float(x) for x in f1],
        per_class_support=[int(x) for x in support],
        confusion=confusion_matrix(targets, preds, labels=labels),
    )


def per_class_report(metrics: Metrics, labels: tuple[str, ...] = EMOTION_LABELS) -> str:
    """Formatted per-class table (string, not printed)."""
    header = f"{'emotion':<10} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>9}"
    lines = [header, "-" * len(header)]
    for i, name in enumerate(labels):
        lines.append(
            f"{name:<10} {metrics.per_class_precision[i]:>10.3f} "
            f"{metrics.per_class_recall[i]:>8.3f} "
            f"{metrics.per_class_f1[i]:>8.3f} "
            f"{metrics.per_class_support[i]:>9d}"
        )
    lines.append("-" * len(header))
    lines.append(f"{'accuracy':<10} {metrics.accuracy:>10.4f}")
    lines.append(f"{'macro_f1':<10} {metrics.macro_f1:>10.4f}")
    return "\n".join(lines)


__all__ = [
    "EMOTION_LABELS",
    "Metrics",
    "Predictions",
    "compute_metrics",
    "load_checkpoint",
    "per_class_report",
    "predict",
]
