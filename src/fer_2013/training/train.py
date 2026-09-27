"""Training loop for FER-2013.

Full fine-tune with AdamW + CosineAnnealingLR.
Early stopping on validation macro-F1 (default) to counter class imbalance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from fer_2013.training.config import Config

log = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    """Per-epoch metrics collected during training."""

    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_score: float = float("-inf")
    best_val_loss: float = float("inf")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (mean_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    n_classes: int = 7,
) -> dict[str, float]:
    """Evaluate on a loader. Returns loss, acc, macro_f1."""
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        total += x.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return {
        "loss": running_loss / total,
        "acc": float((preds == targets).mean()),
        "macro_f1": float(
            f1_score(
                targets,
                preds,
                average="macro",
                labels=list(range(n_classes)),
                zero_division=0,
            )
        ),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_score": best_val_score,
        },
        path,
    )


def _score_is_better(new: float, best: float, mode: str) -> bool:
    if mode == "max":
        return new > best
    return new < best


def fit(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    config: Config,
    *,
    class_weights: torch.Tensor | None = None,
) -> TrainHistory:
    """Run the training loop. Saves best + last checkpoints."""
    tcfg = config.training
    torch.manual_seed(tcfg.seed)
    device = torch.device(config.device)
    model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=tcfg.lr,
        weight_decay=tcfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=tcfg.epochs)

    history = TrainHistory()
    initial_best = float("-inf") if tcfg.early_stopping_mode == "max" else float("inf")
    history.best_val_score = initial_best
    epochs_without_improvement = 0

    for epoch in range(tcfg.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, n_classes=config.model.num_classes
        )
        scheduler.step()

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_metrics["loss"])
        history.val_acc.append(val_metrics["acc"])
        history.val_f1.append(val_metrics["macro_f1"])

        tracked = (
            val_metrics["macro_f1"]
            if tcfg.early_stopping_metric == "macro_f1"
            else val_metrics["loss"]
        )

        log.info(
            "epoch %d/%d  train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f  val_f1=%.4f  lr=%.2e",
            epoch + 1,
            tcfg.epochs,
            train_loss,
            train_acc,
            val_metrics["loss"],
            val_metrics["acc"],
            val_metrics["macro_f1"],
            scheduler.get_last_lr()[0],
        )

        if config.checkpoint.save_last:
            save_checkpoint(
                config.checkpoint.dir / "last.pt",
                model,
                optimizer,
                epoch,
                history.best_val_score,
            )

        if _score_is_better(tracked, history.best_val_score, tcfg.early_stopping_mode):
            history.best_val_score = tracked
            history.best_epoch = epoch
            history.best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            if config.checkpoint.save_best:
                save_checkpoint(
                    config.checkpoint.dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    history.best_val_score,
                )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= tcfg.early_stopping_patience:
            log.info(
                "early stopping at epoch %d (best epoch %d, %s=%.4f)",
                epoch + 1,
                history.best_epoch + 1,
                tcfg.early_stopping_metric,
                history.best_val_score,
            )
            break

    return history


__all__ = [
    "TrainHistory",
    "evaluate",
    "fit",
    "save_checkpoint",
    "train_one_epoch",
]
