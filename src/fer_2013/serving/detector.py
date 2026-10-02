"""Face detection: the Box type, the FaceDetector interface and YuNet on ONNX Runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import onnxruntime as ort
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


STRIDES = (8, 16, 32)


class _Candidate(NamedTuple):
    """A decoded detection in float padded-input pixels, before NMS."""

    x: float
    y: float
    w: float
    h: float
    score: float


def _decode(
    heads: dict[str, np.ndarray], height: int, width: int, threshold: float
) -> list[_Candidate]:
    """Candidate boxes (float, padded-input pixels) from YuNet's cls/obj/bbox heads."""
    found: list[_Candidate] = []
    for stride in STRIDES:
        cols = width // stride
        cls = heads[f"cls_{stride}"].reshape(-1).astype(np.float64)
        obj = heads[f"obj_{stride}"].reshape(-1).astype(np.float64)
        bbox = heads[f"bbox_{stride}"].reshape(-1, 4).astype(np.float64)
        score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        for i in np.nonzero(score >= threshold)[0]:
            row, col = divmod(int(i), cols)
            cx = (col + bbox[i, 0]) * stride
            cy = (row + bbox[i, 1]) * stride
            w = float(np.exp(bbox[i, 2]) * stride)
            h = float(np.exp(bbox[i, 3]) * stride)
            found.append(_Candidate(cx - w / 2, cy - h / 2, w, h, float(score[i])))
    return found


def _iou(a: _Candidate, b: _Candidate) -> float:
    w = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    h = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = w * h
    return inter / (a.w * a.h + b.w * b.h - inter + 1e-9)


def _nms(boxes: list[_Candidate], threshold: float) -> list[_Candidate]:
    """Greedy non-maximum suppression, best score first."""
    kept: list[_Candidate] = []
    for box in sorted(boxes, key=lambda b: b.score, reverse=True):
        if all(_iou(k, box) <= threshold for k in kept):
            kept.append(box)
    return kept


class YuNetDetector:
    """YuNet (MIT) on ONNX Runtime, so the image needs no OpenCV.

    OpenCV decodes the network's raw heads internally; here the decoding and the
    NMS are done in numpy, the same algorithm as `web/src/detector.js`. The
    dynamic-shape model (2026may) runs at the image's own size, zero-padded down
    and right to a multiple of 32. Large images are downscaled to `max_side`
    first and the boxes are mapped back to the original size.
    `tests/serving/test_detector.py` checks the boxes against `cv2.FaceDetectorYN`.
    """

    def __init__(
        self,
        model_path: Path,
        score_threshold: float = 0.6,
        max_side: int = 640,
        max_faces: int | None = None,
        nms_threshold: float = 0.3,
    ) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self._score_threshold = score_threshold
        self._max_side = max_side
        self._max_faces = max_faces
        self._nms_threshold = nms_threshold

    def detect(self, image: Image.Image) -> list[Box]:
        rgb = image.convert("RGB")
        scale = min(1.0, self._max_side / max(rgb.size))
        if scale < 1.0:
            rgb = rgb.resize(
                (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
                Image.Resampling.BILINEAR,
            )
        height = -(-rgb.height // 32) * 32
        width = -(-rgb.width // 32) * 32
        padded = np.zeros((1, 3, height, width), dtype=np.float32)
        padded[0, :, : rgb.height, : rgb.width] = np.asarray(rgb)[:, :, ::-1].transpose(
            2, 0, 1
        )  # BGR
        names = [
            f"{kind}_{stride}" for stride in STRIDES for kind in ("cls", "obj", "bbox")
        ]
        outputs = self._sess.run(names, {"input": padded})
        heads = dict(zip(names, outputs, strict=True))
        boxes = [
            Box(
                int(b.x / scale),
                int(b.y / scale),
                int(b.w / scale),
                int(b.h / scale),
                b.score,
            )
            for b in _nms(
                _decode(heads, height, width, self._score_threshold),
                self._nms_threshold,
            )
        ]
        return boxes if self._max_faces is None else boxes[: self._max_faces]
