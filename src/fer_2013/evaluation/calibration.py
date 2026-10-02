"""Confidence calibration: ECE, reliability bins and temperature scaling.

A model is calibrated when, among predictions made with confidence c, a
fraction c turn out right. Temperature scaling divides the logits by a single
scalar T before the softmax; it fixes over- or under-confidence without
changing which class wins (the argmax is unchanged).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    result: np.ndarray = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return result


def calibrated_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(logits / temperature), stable for large logits."""
    probs: np.ndarray = np.exp(_log_softmax(logits / temperature))
    return probs


def negative_log_likelihood(
    logits: np.ndarray, targets: np.ndarray, temperature: float = 1.0
) -> float:
    """Mean cross-entropy of softmax(logits / temperature)."""
    logp = _log_softmax(logits / temperature)
    return float(-logp[np.arange(len(targets)), targets].mean())


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared distance between the probability vector and the one-hot label."""
    onehot = np.eye(probs.shape[1])[targets]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def reliability_bins(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin mean confidence, accuracy and count over (i/n, (i+1)/n] bins.

    Empty bins have count 0 and NaN confidence and accuracy.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == targets).astype(np.float64)
    idx = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    mean_conf = np.full(n_bins, np.nan)
    accuracy = np.full(n_bins, np.nan)
    for b in np.nonzero(count)[0]:
        mask = idx == b
        mean_conf[b] = conf[mask].mean()
        accuracy[b] = correct[mask].mean()
    return mean_conf, accuracy, count


def expected_calibration_error(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> float:
    """Expected calibration error of the top-label confidence (15 bins)."""
    conf, acc, count = reliability_bins(probs, targets, n_bins)
    filled = count > 0
    gaps = np.abs(acc[filled] - conf[filled])
    return float((count[filled] / count.sum() * gaps).sum())


def fit_temperature(
    logits: np.ndarray, targets: np.ndarray, low: float = 0.05, high: float = 20.0
) -> float:
    """Temperature that minimizes the NLL of softmax(logits / T).

    The NLL is convex in beta = 1 / T, so a golden-section search over beta
    finds the optimum. Fit on a held-out split, never on the test set.
    """

    def nll(beta: float) -> float:
        return negative_log_likelihood(logits, targets, 1.0 / beta)

    a, b = 1.0 / high, 1.0 / low
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = nll(d)
    return 1.0 / ((a + b) / 2)


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw logits (N, C) float32 and integer targets (N,) over a loader."""
    model.eval()
    model.to(device)
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for x, y in loader:
        all_logits.append(model(x.to(device)).float().cpu().numpy())
        all_targets.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets).astype(np.int64)


__all__ = [
    "brier_score",
    "calibrated_probs",
    "collect_logits",
    "expected_calibration_error",
    "fit_temperature",
    "negative_log_likelihood",
    "reliability_bins",
]
