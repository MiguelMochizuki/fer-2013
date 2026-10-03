"""Tests for fer_2013.training.config."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fer_2013.training.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)

# ==============================
# defaults
# ==============================


def test_default_config_has_sensible_values() -> None:
    cfg = Config()
    assert cfg.data.batch_size == 64
    assert cfg.model.num_classes == 7
    assert cfg.training.epochs == 30
    assert cfg.training.early_stopping_metric == "macro_f1"
    assert cfg.training.early_stopping_mode == "max"
    assert cfg.checkpoint.dir == Path("checkpoints")


def test_device_property_returns_string() -> None:
    cfg = Config()
    assert cfg.device in {"cuda", "cpu"}


# ==============================
# validation
# ==============================


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        DataConfig(batch_size=0)


def test_num_workers_can_be_zero() -> None:
    cfg = DataConfig(num_workers=0)
    assert cfg.data if False else cfg.num_workers == 0  # keep ruff happy


def test_lr_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(lr=0.0)


def test_early_stopping_metric_must_be_known() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(early_stopping_metric="accuracy")  # type: ignore[arg-type]


def test_mode_must_match_metric() -> None:
    """macro_f1 requires mode='max'."""
    with pytest.raises(ValidationError):
        TrainingConfig(early_stopping_metric="macro_f1", early_stopping_mode="min")


def test_loss_requires_min_mode() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(early_stopping_metric="loss", early_stopping_mode="max")


def test_loss_with_min_mode_is_valid() -> None:
    cfg = TrainingConfig(early_stopping_metric="loss", early_stopping_mode="min")
    assert cfg.early_stopping_metric == "loss"
    assert cfg.early_stopping_mode == "min"


# ==============================
# load_config
# ==============================


def test_load_config_missing_path_returns_defaults() -> None:
    cfg = load_config(None)
    assert cfg.training.epochs == 30


def test_load_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("training:\n  epochs: 5\n  lr: 0.001\ndata:\n  batch_size: 32\n")
    cfg = load_config(path)
    assert cfg.training.epochs == 5
    assert cfg.training.lr == 0.001
    assert cfg.data.batch_size == 32
    # Fields not in the YAML keep their defaults
    assert cfg.model.num_classes == 7


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    """Pydantic BaseModel default: extra fields forbidden."""
    path = tmp_path / "config.yaml"
    path.write_text("training:\n  typo_field: 5\n")
    with pytest.raises(ValidationError):
        load_config(path)


# ==============================
# composition
# ==============================


def test_nested_config_accepts_sub_configs() -> None:
    cfg = Config(
        data=DataConfig(batch_size=8, num_workers=0),
        model=ModelConfig(num_classes=3, pretrained=False),
        training=TrainingConfig(epochs=2),
        checkpoint=CheckpointConfig(dir=Path("/tmp/ckpt")),
    )
    assert cfg.data.batch_size == 8
    assert cfg.model.num_classes == 3
    assert cfg.training.epochs == 2
    assert cfg.checkpoint.dir == Path("/tmp/ckpt")


def test_balanced_sampler_is_on_by_default_and_overridable() -> None:
    assert Config().data.balanced_sampler is True
    assert (
        Config.model_validate(
            {"data": {"balanced_sampler": False}}
        ).data.balanced_sampler
        is False
    )


def test_class_weights_are_on_by_default_and_overridable() -> None:
    assert Config().training.class_weights is True
    cfg = Config.model_validate({"training": {"class_weights": False}})
    assert cfg.training.class_weights is False
