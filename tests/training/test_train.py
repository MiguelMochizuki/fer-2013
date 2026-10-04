"""Tests for fer_2013.training.train."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from fer_2013.training.config import (
    CheckpointConfig,
    Config,
    DataConfig,
    ModelConfig,
    TensorBoardConfig,
    TrainingConfig,
)
from fer_2013.training.train import (
    TrainHistory,
    evaluate,
    fit,
    save_checkpoint,
    train_one_epoch,
)

Loader = DataLoader[tuple[torch.Tensor, torch.Tensor]]
PairDataset = Dataset[tuple[torch.Tensor, torch.Tensor]]


@pytest.fixture
def tiny_loaders() -> tuple[Loader, Loader]:
    """Two tiny loaders with random 3x16x16 inputs and 7 classes."""
    g = torch.Generator().manual_seed(0)
    x_train = torch.randn(16, 3, 16, 16, generator=g)
    y_train = torch.randint(0, 7, (16,), generator=g)
    x_val = torch.randn(8, 3, 16, 16, generator=g)
    y_val = torch.randint(0, 7, (8,), generator=g)

    train_ds = cast(PairDataset, TensorDataset(x_train, y_train))
    val_ds = cast(PairDataset, TensorDataset(x_val, y_val))

    train_loader: Loader = DataLoader(train_ds, batch_size=8)
    val_loader: Loader = DataLoader(val_ds, batch_size=4)
    return train_loader, val_loader


class TinyNet(nn.Module):
    """Fast stand-in for ResNet18."""

    def __init__(self, n_classes: int = 7) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled: torch.Tensor = self.pool(x).flatten(1)
        out: torch.Tensor = self.fc(pooled)
        return out


def _make_config(
    tmp_path: Path,
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    patience: int = 10,
    metric: Literal["macro_f1", "loss"] = "macro_f1",
    mode: Literal["max", "min"] = "max",
    tensorboard_enabled: bool = False,
) -> Config:
    """Build a minimal Config suitable for tests.

    TensorBoard is disabled by default so tests don't write to ./runs.
    Tests that exercise it opt in and point log_dir at tmp_path.
    """
    return Config(
        data=DataConfig(batch_size=8, num_workers=0),
        model=ModelConfig(num_classes=7, pretrained=False),
        training=TrainingConfig(
            epochs=epochs,
            lr=lr,
            early_stopping_patience=patience,
            early_stopping_metric=metric,
            early_stopping_mode=mode,
        ),
        checkpoint=CheckpointConfig(dir=tmp_path / "ckpt"),
        tensorboard=TensorBoardConfig(
            enabled=tensorboard_enabled,
            log_dir=tmp_path / "runs",
            run_name="test",
        ),
    )


# ==============================
# train_one_epoch
# ==============================


def test_train_one_epoch_returns_loss_and_accuracy(
    tiny_loaders: tuple[Loader, Loader],
) -> None:
    train_loader, _ = tiny_loaders
    model = TinyNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss, acc, f1, step = train_one_epoch(
        model, train_loader, criterion, optimizer, torch.device("cpu")
    )
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert step == len(train_loader)


def test_train_one_epoch_updates_parameters(
    tiny_loaders: tuple[Loader, Loader],
) -> None:
    train_loader, _ = tiny_loaders
    model = TinyNet()
    before = model.fc.weight.detach().clone()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_one_epoch(model, train_loader, criterion, optimizer, torch.device("cpu"))
    after = model.fc.weight.detach()
    assert not torch.equal(before, after)


# ==============================
# evaluate
# ==============================


def test_evaluate_returns_loss_acc_f1(
    tiny_loaders: tuple[Loader, Loader],
) -> None:
    _, val_loader = tiny_loaders
    model = TinyNet()
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, val_loader, criterion, torch.device("cpu"), n_classes=7)
    assert set(metrics.keys()) == {"loss", "acc", "macro_f1"}
    assert metrics["loss"] >= 0.0
    assert 0.0 <= metrics["acc"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_evaluate_does_not_update_parameters(
    tiny_loaders: tuple[Loader, Loader],
) -> None:
    _, val_loader = tiny_loaders
    model = TinyNet()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    criterion = nn.CrossEntropyLoss()
    evaluate(model, val_loader, criterion, torch.device("cpu"))
    after = model.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k])


# ==============================
# save_checkpoint
# ==============================


def test_save_checkpoint_creates_file(tmp_path: Path) -> None:
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "ckpt" / "best.pt"
    save_checkpoint(path, model, optimizer, epoch=3, best_val_score=0.42)
    assert path.exists()
    loaded = torch.load(path, weights_only=False)
    assert set(loaded.keys()) == {
        "model",
        "optimizer",
        "epoch",
        "best_val_score",
    }
    assert loaded["epoch"] == 3
    assert loaded["best_val_score"] == 0.42


# ==============================
# fit
# ==============================


def test_fit_runs_and_returns_history(
    tiny_loaders: tuple[Loader, Loader],
    tmp_path: Path,
) -> None:
    train_loader, val_loader = tiny_loaders
    model = TinyNet()
    config = _make_config(tmp_path, epochs=3)
    history = fit(
        model, train_loader, val_loader, config, reports_dir=tmp_path / "reports"
    )
    assert isinstance(history, TrainHistory)
    assert len(history.train_loss) == 3
    assert len(history.val_loss) == 3
    assert history.best_epoch >= 0
    assert history.best_val_score > float("-inf")
    assert (tmp_path / "ckpt" / "best.pt").exists()
    assert (tmp_path / "ckpt" / "last.pt").exists()
    assert any((tmp_path / "reports").glob("history_*.json"))


def test_fit_early_stops(
    tiny_loaders: tuple[Loader, Loader],
    tmp_path: Path,
) -> None:
    """With tiny LR and patience=1, val score won't improve past epoch 1."""
    train_loader, val_loader = tiny_loaders
    model = TinyNet()
    config = _make_config(tmp_path, epochs=20, lr=1e-12, patience=1)
    history = fit(
        model, train_loader, val_loader, config, reports_dir=tmp_path / "reports"
    )
    assert len(history.train_loss) < 5


def test_fit_accepts_class_weights(
    tiny_loaders: tuple[Loader, Loader],
    tmp_path: Path,
) -> None:
    train_loader, val_loader = tiny_loaders
    model = TinyNet()
    config = _make_config(tmp_path, epochs=1)
    weights = torch.ones(7)
    history = fit(
        model,
        train_loader,
        val_loader,
        config,
        class_weights=weights,
        reports_dir=tmp_path / "reports",
    )
    assert len(history.train_loss) == 1


def test_fit_can_stop_on_loss(
    tiny_loaders: tuple[Loader, Loader],
    tmp_path: Path,
) -> None:
    """Flip to loss/min and confirm the loop still runs."""
    train_loader, val_loader = tiny_loaders
    model = TinyNet()
    config = _make_config(tmp_path, epochs=2, metric="loss", mode="min", patience=5)
    history = fit(
        model, train_loader, val_loader, config, reports_dir=tmp_path / "reports"
    )
    assert len(history.train_loss) == 2
    assert history.best_val_score < float("inf")


def test_fit_writes_history_json(
    tiny_loaders: tuple[Loader, Loader],
    tmp_path: Path,
) -> None:
    train_loader, val_loader = tiny_loaders
    model = TinyNet()
    config = _make_config(tmp_path, epochs=1)
    reports_dir = tmp_path / "reports"
    fit(model, train_loader, val_loader, config, reports_dir=reports_dir)
    files = list(reports_dir.glob("history_*.json"))
    assert len(files) == 1


def test_fit_writes_tensorboard_logs(
    tiny_loaders: tuple[Loader, Loader],
    tmp_path: Path,
) -> None:
    """TensorBoard log dir must be created when enabled."""
    train_loader, val_loader = tiny_loaders
    model = TinyNet()
    config = _make_config(tmp_path, epochs=1, tensorboard_enabled=True)
    fit(model, train_loader, val_loader, config, reports_dir=tmp_path / "reports")
    assert (tmp_path / "runs").exists()
    assert any((tmp_path / "runs").rglob("events.out.tfevents.*"))


def test_history_json_records_provenance(tmp_path: Path) -> None:
    import json

    from fer_2013.training.train import TrainHistory, save_history

    path = save_history(TrainHistory(), tmp_path, {"git": "abc123", "config": {}})
    assert json.loads(path.read_text())["provenance"]["git"] == "abc123"


def test_same_seed_builds_the_same_model_head() -> None:
    from fer_2013.models.cnn import build_resnet18
    from fer_2013.training.train import seed_everything

    def head() -> torch.Tensor:
        model = build_resnet18(num_classes=7, pretrained=False)
        return torch.cat([p.flatten() for p in model.fc.parameters()])

    seed_everything(42)
    first = head()
    seed_everything(42)
    assert torch.equal(first, head())
