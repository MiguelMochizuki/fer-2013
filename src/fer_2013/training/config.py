"""Typed configuration for training runs.

Loads YAML, validates, returns a nested Pydantic model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StrictBase(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(_StrictBase):
    """Where the preprocessed arrays are and how they are batched."""

    processed_dir: Path = Path("data/processed")
    batch_size: int = Field(default=64, ge=1)
    num_workers: int = Field(default=4, ge=0)


class ModelConfig(_StrictBase):
    """ResNet18 head size and whether to start from ImageNet weights."""

    num_classes: int = Field(default=7, ge=2)
    pretrained: bool = True


class TrainingConfig(_StrictBase):
    """Optimizer, early stopping and seed. The stopping metric and its mode must agree."""

    epochs: int = Field(default=30, ge=1)
    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    early_stopping_patience: int = Field(default=5, ge=0)
    early_stopping_metric: Literal["macro_f1", "loss"] = "macro_f1"
    early_stopping_mode: Literal["max", "min"] = "max"
    seed: int = 42

    @field_validator("early_stopping_mode")
    @classmethod
    def _check_mode_matches_metric(cls, v: str, info: object) -> str:
        metric = getattr(info, "data", {}).get("early_stopping_metric")
        if metric == "macro_f1" and v != "max":
            raise ValueError("early_stopping_mode must be 'max' for metric 'macro_f1'")
        if metric == "loss" and v != "min":
            raise ValueError("early_stopping_mode must be 'min' for metric 'loss'")
        return v


class CheckpointConfig(_StrictBase):
    """Where checkpoints go and which ones to keep."""

    dir: Path = Path("checkpoints")
    save_best: bool = True
    save_last: bool = True


class TensorBoardConfig(_StrictBase):
    """TensorBoard logging switches."""

    enabled: bool = True
    log_dir: Path = Path("runs")
    run_name: str = "fer2013_resnet18"


class Config(_StrictBase):
    """The whole training configuration; unknown keys are rejected."""

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    tensorboard: TensorBoardConfig = Field(default_factory=TensorBoardConfig)

    @property
    def device(self) -> str:
        """``cuda`` when a GPU is available, otherwise ``cpu``."""
        return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: Path | None = None) -> Config:
    """Load a YAML config file. Missing path → default config."""
    if path is None:
        return Config()
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)


__all__ = [
    "CheckpointConfig",
    "Config",
    "DataConfig",
    "ModelConfig",
    "TensorBoardConfig",
    "TrainingConfig",
    "load_config",
]
