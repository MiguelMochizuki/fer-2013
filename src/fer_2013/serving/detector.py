"""Face detection: the Box type and the FaceDetector interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """Axis-aligned face box in pixels of the original image."""

    x: int
    y: int
    w: int
    h: int
    score: float
