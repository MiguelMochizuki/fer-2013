"""Thin wrapper around pytorch-grad-cam for ResNet18.

Uses GradCAM++ by default; the underlying library also provides
GradCAM, ScoreCAM, AblationCAM, EigenCAM, etc.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn

CAMMethod = Literal["gradcam", "gradcam++"]


class _ClassifierTarget:
    """Targets the logit of a given class index. Compatible with pytorch-grad-cam.

    Handles both batched outputs (N, C) and single-sample outputs (C,), because
    the library sometimes calls the target per-sample after indexing.
    """

    def __init__(self, class_idx: int) -> None:
        self.class_idx = class_idx

    def __call__(self, output: torch.Tensor) -> torch.Tensor:
        if output.dim() == 2:
            return output[:, self.class_idx]
        return output[self.class_idx]


def find_target_layer(model: nn.Module) -> nn.Module:
    """Return the last conv block of a torchvision ResNet."""
    if not hasattr(model, "layer4"):
        raise AttributeError("Expected a torchvision ResNet with a `layer4` attribute.")
    layer4 = cast(nn.Sequential, model.layer4)
    return layer4[-1]


def generate_heatmaps(
    model: nn.Module,
    images: torch.Tensor,
    class_indices: list[int],
    *,
    method: CAMMethod = "gradcam++",
    target_layer: nn.Module | None = None,
) -> np.ndarray:
    """Generate one heatmap per image for the given class indices.

    Args:
        model: ResNet-like classifier.
        images: (N, 3, H, W) float tensor on the same device as model.
        class_indices: length-N list of class indices to explain.
        method: "gradcam" or "gradcam++".
        target_layer: optional override; defaults to model.layer4[-1].

    Returns:
        (N, H, W) float32 heatmaps in [0, 1].
    """
    if target_layer is None:
        target_layer = find_target_layer(model)

    cam_cls = GradCAM if method == "gradcam" else GradCAMPlusPlus

    # Ensure model and inputs are on the same device, in eval mode.
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)
    targets = [_ClassifierTarget(c) for c in class_indices]

    with cam_cls(model=model, target_layers=[target_layer]) as cam:
        grayscale_cams = cam(input_tensor=images, targets=targets)

    if grayscale_cams is None:
        raise RuntimeError(
            "pytorch-grad-cam returned None. Check that model is in eval mode, "
            "images are on the same device as the model, and targets match "
            "the batch size."
        )

    return np.asarray(grayscale_cams, dtype=np.float32)


def overlay(
    image_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Blend a (H, W, 3) uint8 image with a (H, W) heatmap.

    Returns (H, W, 3) uint8.
    """
    image_float = image_uint8.astype(np.float32) / 255.0
    return np.asarray(
        show_cam_on_image(image_float, heatmap, use_rgb=True, image_weight=1 - alpha)
    )


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Convert an ImageNet-normalized (3, H, W) tensor to (H, W, 3) uint8."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = tensor.detach().cpu() * std_t + mean_t
    img = img.clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


__all__ = [
    "CAMMethod",
    "denormalize",
    "find_target_layer",
    "generate_heatmaps",
    "overlay",
]
