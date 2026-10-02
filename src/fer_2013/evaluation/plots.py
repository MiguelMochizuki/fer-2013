"""Plotting helpers for FER-2013 evaluation.

Pure functions: take arrays/history dicts, write PNGs, return the path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve

from fer_2013.evaluation.evaluate import EMOTION_LABELS

sns.set_theme(style="whitegrid", context="notebook")

DEFAULT_DPI = 120


class HistoryDict(TypedDict):
    """Schema of the JSON written by training.save_history."""

    train_loss: list[float]
    train_acc: list[float]
    train_f1: list[float]
    val_loss: list[float]
    val_acc: list[float]
    val_f1: list[float]
    best_epoch: int
    best_val_score: float


def plot_training_curves(
    history: HistoryDict,
    out_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Loss and macro-F1 curves for train and val.

    Args:
        history: dict with keys train_loss, val_loss, train_f1, val_f1, best_epoch.
        out_path: where to write the PNG.
        dpi: figure resolution.

    Returns:
        out_path.
    """
    best_epoch = history["best_epoch"] + 1
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train", marker="o", ms=3)
    axes[0].plot(epochs, history["val_loss"], label="val", marker="o", ms=3)
    axes[0].axvline(
        best_epoch,
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"best epoch ({best_epoch})",
    )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss, train vs val")
    axes[0].legend()

    axes[1].plot(epochs, history["train_f1"], label="train", marker="o", ms=3)
    axes[1].plot(epochs, history["val_f1"], label="val", marker="o", ms=3)
    axes[1].axvline(
        best_epoch,
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"best epoch ({best_epoch})",
    )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("macro-F1")
    axes[1].set_title("Macro-F1, train vs val")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_reliability_diagram(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Accuracy against confidence, one curve per entry (from `reliability_bins`).

    Points on the diagonal are perfectly calibrated; below it, over-confident.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    for name, (conf, acc, count) in curves.items():
        filled = count > 0
        ax.plot(conf[filled], acc[filled], marker="o", label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    confusion: np.ndarray,
    out_path: Path,
    *,
    labels: tuple[str, ...] = EMOTION_LABELS,
    normalize: bool = True,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Heatmap of the confusion matrix. Row-normalized by default."""
    cm = confusion.astype(np.float64)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1 if normalize else None,
        cbar_kws={"label": "fraction of true class" if normalize else "count"},
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Normalized confusion matrix" if normalize else "Confusion matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_pr_curves(
    probs: np.ndarray,
    targets: np.ndarray,
    out_path: Path,
    *,
    labels: tuple[str, ...] = EMOTION_LABELS,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """One-vs-rest precision-recall curve per class."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("Set2", len(labels))

    for i, (label, color) in enumerate(zip(labels, colors, strict=True)):
        y_true = (targets == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, probs[:, i])
        ax.plot(recall, precision, label=label, color=color)

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall per class (one-vs-rest)")
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_image_grid(
    images: list[np.ndarray],
    titles: list[str],
    out_path: Path,
    *,
    ncols: int = 4,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Grid of (H, W, 3) uint8 images, one per subplot, titled.

    Args:
        images: list of (H, W, 3) uint8 arrays.
        titles: one title per image, same length as images.
        out_path: where to write the PNG.
        ncols: number of columns; row count follows from len(images).
        dpi: figure resolution.

    Returns:
        out_path.
    """
    n = len(images)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)  # ceil division

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    flat_axes = np.atleast_1d(axes).flatten()

    for ax, image, title in zip(flat_axes, images, titles, strict=False):
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in flat_axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


__all__ = [
    "HistoryDict",
    "plot_confusion_matrix",
    "plot_image_grid",
    "plot_pr_curves",
    "plot_training_curves",
]
