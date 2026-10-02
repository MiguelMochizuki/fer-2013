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
    """Convert to 8-bit grayscale, scaling 16-bit and float images to the full 0-255 range."""
    if img.mode.startswith("I") or img.mode == "F":
        a = np.asarray(img, dtype=np.float32)
        hi = 65535.0 if img.mode.startswith("I;16") else max(float(a.max()), 1.0)
        return Image.fromarray((a / hi * 255).astype(np.uint8), "L")
    return img.convert("L")


def _fit(start: int, side: int, limit: int) -> int:
    """Slide a window of `side` into [0, limit]; if it cannot fit, start at 0."""
    return 0 if side >= limit else min(max(start, 0), limit - side)


def crop_gray(image: Image.Image, box: Box, margin: float = 0.10) -> Image.Image | None:
    """Square crop around `box` (grown by `margin` on every side) as an 'L' image.

    FER-2013 faces are square, so the crop is too; near a border the square is
    slid inside the image. Returns None if the box is empty or outside it.
    """
    if box.w <= 0 or box.h <= 0:
        return None
    if box.x + box.w <= 0 or box.x >= image.width:
        return None
    if box.y + box.h <= 0 or box.y >= image.height:
        return None
    side = int(max(box.w, box.h) * (1 + 2 * margin))
    x0 = _fit(int(box.x + box.w / 2 - side / 2), side, image.width)
    y0 = _fit(int(box.y + box.h / 2 - side / 2), side, image.height)
    crop = image.crop(
        (x0, y0, min(x0 + side, image.width), min(y0 + side, image.height))
    )
    return to_gray(crop)


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
