"""Face detection: the Box type and the FaceDetector interface."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Box:
    """Axis-aligned face box in pixels of the original image."""

    x: int
    y: int
    w: int
    h: int
    score: float


class FaceDetector(Protocol):
    """Anything that finds faces in an RGB image. Boxes are in image pixels."""

    def detect(self, image: Image.Image) -> list[Box]: ...


class YuNetDetector:
    """OpenCV YuNet (MIT) via cv2.FaceDetectorYN.

    setInputSize mutates the detector, so detect() is serialized with a lock.
    Large images are downscaled to `max_side` before detection and the boxes
    are mapped back to the original size.
    """

    def __init__(
        self,
        model_path: Path,
        score_threshold: float = 0.6,
        max_side: int = 640,
        max_faces: int = 10,
    ) -> None:
        self._det = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold
        )
        self._max_side = max_side
        self._max_faces = max_faces
        self._lock = threading.Lock()

    def detect(self, image: Image.Image) -> list[Box]:
        rgb = image.convert("RGB")
        scale = min(1.0, self._max_side / max(rgb.size))
        if scale < 1.0:
            rgb = rgb.resize(
                (round(rgb.width * scale), round(rgb.height * scale)),
                Image.Resampling.BILINEAR,
            )
        bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
        with self._lock:
            self._det.setInputSize((rgb.width, rgb.height))
            _, faces = self._det.detect(bgr)
        if faces is None:
            return []
        boxes = [
            Box(
                int(f[0] / scale),
                int(f[1] / scale),
                int(f[2] / scale),
                int(f[3] / scale),
                float(f[-1]),
            )
            for f in faces
        ]
        boxes.sort(key=lambda b: b.score, reverse=True)
        return boxes[: self._max_faces]
