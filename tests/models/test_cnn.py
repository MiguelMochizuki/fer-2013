"""Tests for fer_2013.models.cnn."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torchvision.models import ResNet

from fer_2013.models.cnn import (
    FEATURE_DIM,
    NUM_CLASSES,
    build_resnet18,
    count_trainable_params,
    freeze_backbone,
    unfreeze_all,
)


@pytest.fixture
def model() -> ResNet:
    """Fresh, untrained ResNet18 with 7-class head per test."""
    return build_resnet18(pretrained=False)


# ==============================
# build_resnet18
# ==============================


def test_output_shape(model: ResNet) -> None:
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_head_is_fresh(model: ResNet) -> None:
    """The fc layer must be a fresh Linear(512, 7), not the ImageNet 1000-class one."""
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == FEATURE_DIM
    assert model.fc.out_features == NUM_CLASSES


def test_num_classes_is_configurable() -> None:
    m = build_resnet18(num_classes=3, pretrained=False)
    assert m.fc.out_features == 3
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 3)


def test_pretrained_false_skips_download() -> None:
    """No network access; must succeed offline."""
    m = build_resnet18(pretrained=False)
    assert isinstance(m, nn.Module)


# ==============================
# freeze_backbone
# ==============================


def test_freeze_backbone_keeps_fc_trainable(model: ResNet) -> None:
    freeze_backbone(model)
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            assert param.requires_grad, f"{name} should be trainable"


def test_freeze_backbone_keeps_bn_trainable(model: ResNet) -> None:
    freeze_backbone(model)
    bn_params = [
        (name, p)
        for name, p in model.named_parameters()
        if ".bn" in name or name.startswith("bn")
    ]
    assert bn_params, "ResNet18 must have BatchNorm params"
    for name, param in bn_params:
        assert param.requires_grad, f"{name} (BN) should stay trainable"


def test_freeze_backbone_freezes_conv_and_other_linear(model: ResNet) -> None:
    freeze_backbone(model)
    for name, param in model.named_parameters():
        if name.startswith("fc.") or ".bn" in name or name.startswith("bn"):
            continue
        assert not param.requires_grad, f"{name} should be frozen"


def test_freeze_backbone_reduces_trainable_params(model: ResNet) -> None:
    total = count_trainable_params(model)
    freeze_backbone(model)
    frozen = count_trainable_params(model)
    assert frozen < total
    # Only fc (512*7 + 7) plus BN params should be trainable.
    assert frozen < total / 4  # sanity: at least 4x reduction


# ==============================
# unfreeze_all
# ==============================


def test_unfreeze_all_restores_everything(model: ResNet) -> None:
    freeze_backbone(model)
    unfreeze_all(model)
    for param in model.parameters():
        assert param.requires_grad


def test_unfreeze_all_restores_param_count(model: ResNet) -> None:
    initial = count_trainable_params(model)
    freeze_backbone(model)
    unfreeze_all(model)
    assert count_trainable_params(model) == initial
