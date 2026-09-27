"""ResNet18 for FER-2013.

ImageNet-pretrained backbone, fresh 7-class head.
Input expected: (B, 3, 224, 224) float32, ImageNet-normalized.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet, ResNet18_Weights, resnet18

NUM_CLASSES = 7
FEATURE_DIM = 512  # ResNet18's final pooled feature dimension


def build_resnet18(
    num_classes: int = NUM_CLASSES,
    *,
    pretrained: bool = True,
) -> ResNet:
    """ResNet18 with a fresh linear head for `num_classes` outputs.

    Args:
        num_classes: number of output logits.
        pretrained: if True, load ImageNet weights for the backbone.

    Returns:
        A ResNet18 whose `fc` layer is freshly initialized.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model: ResNet = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the final `fc` layer.

    BatchNorm running stats and affine params stay trainable so they
    adapt to the target domain.
    """
    for name, param in model.named_parameters():
        if name.startswith("fc.") or ".bn" in name or name.startswith("bn"):
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Set requires_grad=True on every parameter."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    """Return the number of parameters with requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "FEATURE_DIM",
    "NUM_CLASSES",
    "build_resnet18",
    "count_trainable_params",
    "freeze_backbone",
    "unfreeze_all",
]
