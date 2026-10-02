#!/usr/bin/env python3
"""CLI to train ResNet18 on FER-2013.

Usage:
    uv run python scripts/train.py --config configs/default.yaml
    uv run python scripts/train.py --config configs/default.yaml --set training.epochs=5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from fer_2013.data.datamodule import build_class_weights, make_dataloader
from fer_2013.models.cnn import build_resnet18
from fer_2013.training.config import Config, load_config
from fer_2013.training.train import fit


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 on FER-2013.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file. Missing → built-in defaults.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help=(
            "Override a config field (dotted path), e.g. "
            "--set training.epochs=10 --set data.batch_size=128. "
            "Repeatable."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _coerce(value: str) -> Any:
    """Turn '10' → int, '1e-4' → float, 'true' → bool, otherwise str."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_override(cfg: dict[str, Any], dotted: str) -> None:
    key, _, raw_value = dotted.partition("=")
    if not _:
        raise ValueError(f"Invalid override (expected key=value): {dotted!r}")
    keys = key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = _coerce(raw_value)


def _load_with_overrides(config_path: Path | None, overrides: list[str]) -> Config:
    cfg = Config() if config_path is None else load_config(config_path)
    if overrides:
        raw = cfg.model_dump()
        for ov in overrides:
            _apply_override(raw, ov)
        cfg = Config.model_validate(raw)
    return cfg


def main(argv: list[str] | None = None) -> int:
    """Run the training command line tool.

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
    log = logging.getLogger("train")

    config = _load_with_overrides(args.config, args.set)
    log.info("device: %s", config.device)
    log.info("config: %s", config.model_dump())

    train_loader = make_dataloader(
        config.data.processed_dir,
        "train",
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )
    val_loader = make_dataloader(
        config.data.processed_dir,
        "val",
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    y_train = np.load(config.data.processed_dir / "y_train.npy")
    class_weights = build_class_weights(y_train, n_classes=config.model.num_classes)

    model = build_resnet18(
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
    )

    history = fit(model, train_loader, val_loader, config, class_weights=class_weights)

    log.info(
        "training done. best epoch=%d, best %s=%.4f",
        history.best_epoch + 1,
        config.training.early_stopping_metric,
        history.best_val_score,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
