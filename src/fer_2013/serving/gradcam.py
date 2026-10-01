"""Grad-CAM++ without torch, in closed form for the ResNet18 head.

The head is avgpool -> Dropout -> Linear. In eval mode the gradient of a class
logit w.r.t. the layer4 activations is therefore constant over space:
g_k = w_k / (H*W). Plugging that into the Grad-CAM++ weights used by
pytorch-grad-cam gives a closed form that needs only the activations and the
Linear weights. Valid for this head only; a test checks it against
pytorch-grad-cam.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

EPS = 1e-6  # same epsilon as pytorch-grad-cam's GradCAMPlusPlus
NORM_EPS = 1e-7  # same as pytorch-grad-cam's scale_cam_image


def gradcam_pp(
    features: np.ndarray, fc_weight: np.ndarray, class_idx: int
) -> np.ndarray:
    """Grad-CAM++ map for one image and one class.

    Args:
        features: (K, H, W) layer4 activations.
        fc_weight: (num_classes, K) weights of the final Linear layer.
        class_idx: class whose logit is explained.

    Returns:
        (H, W) float32 map in [0, 1].
    """
    k, h, w = features.shape
    hw = h * w
    g = fc_weight[class_idx].astype(np.float32) / hw  # (K,)
    g2, g3 = g**2, g**3
    s = features.sum(axis=(1, 2))  # (K,)
    aij = np.where(g != 0, g2 / (2 * g2 + s * g3 + EPS), 0.0)
    weights = hw * np.maximum(g, 0) * aij  # sum over space of a constant
    cam = np.maximum(np.tensordot(weights, features, axes=1), 0)
    cam = cam - cam.min()
    out: np.ndarray = (cam / (NORM_EPS + cam.max())).astype(np.float32)
    return out


def _jet(x: np.ndarray) -> np.ndarray:
    """Jet-like colormap, x in [0,1] -> (..., 3) floats in [0,1]."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_png_b64(gray: Image.Image, cam: np.ndarray, size: int = 128) -> str:
    """Blend the heatmap over the face crop and return it as a base64 PNG."""
    cam_img = Image.fromarray(cam.astype(np.float32), "F").resize(
        (size, size), Image.Resampling.BILINEAR
    )
    heat = _jet(np.clip(np.asarray(cam_img), 0, 1))
    base = np.asarray(gray.resize((size, size)), dtype=np.float32) / 255.0
    blend = 0.5 * base[..., None] + 0.5 * heat
    out = Image.fromarray((blend * 255).astype(np.uint8), "RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
