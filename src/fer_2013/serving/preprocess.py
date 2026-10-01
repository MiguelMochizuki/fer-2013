"""Face crop -> classifier input, mirroring the training pipeline.

The model was trained on 48x48 grayscale faces that build_transforms("test")
upsamples to 224x224 (bicubic), replicates to 3 channels and normalizes with
ImageNet statistics. The same steps are reproduced here with numpy and Pillow.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from fer_2013.serving.detector import Box

RESIZE_TO = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FACE_SIZE = (48, 48)


def to_gray(img: Image.Image) -> Image.Image:
    if img.mode.startswith("I") or img.mode == "F":
        a = np.asarray(img, dtype=np.float32)
        hi = 65535.0 if img.mode.startswith("I;16") else max(float(a.max()), 1.0)
        return Image.fromarray((a / hi * 255).astype(np.uint8), "L")
    return img.convert("L")


def crop_gray(image: Image.Image, box: Box, margin: float = 0.10) -> Image.Image | None:
    """Crop `box` (plus `margin` on every side) as an 'L' image.

    The crop is clamped to the image. Returns None if it has no area.
    """
    dx, dy = box.w * margin, box.h * margin
    x0 = max(int(box.x - dx), 0)
    y0 = max(int(box.y - dy), 0)
    x1 = min(int(box.x + box.w + dx), image.width)
    y1 = min(int(box.y + box.h + dy), image.height)
    if x1 <= x0 or y1 <= y0:
        return None
    return to_gray(image.crop((x0, y0, x1, y1)))


def to_input(gray: Image.Image) -> np.ndarray:
    """Grayscale face -> float32 (1, 3, 224, 224), ImageNet-normalized."""
    if gray.size != FACE_SIZE:
        gray = gray.resize(FACE_SIZE, Image.Resampling.LANCZOS)
    x = np.asarray(gray, dtype=np.float32) / 255.0
    big = Image.fromarray(x, "F").resize(RESIZE_TO, Image.Resampling.BICUBIC)
    plane = np.asarray(big, dtype=np.float32)
    mean = np.asarray(MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(STD, dtype=np.float32)[:, None, None]
    out = (np.broadcast_to(plane, (3, *plane.shape)) - mean) / std
    return out[None].astype(np.float32)
