import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fer_2013.serving.detector import Box, YuNetDetector

MODEL = Path("models/face_detection_yunet_2026may.onnx")
REFERENCE = Path(
    "models/face_detection_yunet_2023mar.onnx"
)  # OpenCV's own YuNet, the independent reference
FACE = Path(__file__).parent / "fixtures" / "face.jpg"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="needs YuNet model")


@pytest.fixture(scope="module")
def detector() -> YuNetDetector:
    return YuNetDetector(MODEL)


def _center(b: Box) -> tuple[float, float]:
    return (b.x + b.w / 2, b.y + b.h / 2)


def test_detects_nothing_on_blank_image(detector: YuNetDetector) -> None:
    assert detector.detect(Image.new("RGB", (300, 300), (128, 128, 128))) == []


def test_detects_the_fixture_face(detector: YuNetDetector) -> None:
    boxes = detector.detect(Image.open(FACE).convert("RGB"))
    assert len(boxes) >= 1
    cx, cy = _center(boxes[0])
    assert abs(cx - 125) < 50 and abs(cy - 95) < 50  # face position in the crop
    assert 0 < boxes[0].score <= 1


def test_boxes_scaled_to_original_size(detector: YuNetDetector) -> None:
    base = Image.open(FACE).convert("RGB")
    big = base.resize((1300, 1300))  # larger than max_side=640
    boxes = detector.detect(big)
    assert len(boxes) >= 1
    f = 1300 / base.width
    cx, cy = _center(boxes[0])
    assert abs(cx - 125 * f) < 50 * f and abs(cy - 95 * f) < 50 * f
    assert boxes[0].x >= 0 and boxes[0].x + boxes[0].w <= 1300 * 1.05


def test_returns_at_most_max_faces() -> None:
    face = Image.open(FACE).convert("RGB")
    grid = Image.new("RGB", (2 * face.width, 2 * face.height))
    for i in range(2):
        for j in range(2):
            grid.paste(face, (i * face.width, j * face.height))
    assert len(YuNetDetector(MODEL, max_faces=2).detect(grid)) == 2
    assert len(YuNetDetector(MODEL, max_faces=10).detect(grid)) >= 3


def test_default_does_not_cap_the_number_of_faces(detector: YuNetDetector) -> None:
    face = Image.open(FACE).convert("RGB")
    n = 4
    grid = Image.new("RGB", (n * face.width, n * face.height))
    for i in range(n):
        for j in range(n):
            grid.paste(face, (i * face.width, j * face.height))
    assert len(detector.detect(grid)) > 10


def test_concurrent_calls_do_not_corrupt(detector: YuNetDetector) -> None:
    img = Image.open(FACE).convert("RGB")
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: detector.detect(img), range(12)))
    assert all(len(r) >= 1 for r in results)
    first = results[0][0]
    assert all(r[0] == first for r in results)


@pytest.mark.parametrize("size", [(5000, 2), (20000, 1)])
def test_very_thin_images_do_not_crash(
    detector: YuNetDetector, size: tuple[int, int]
) -> None:
    assert detector.detect(Image.new("RGB", size)) == []


def test_serving_does_not_import_opencv() -> None:
    code = "import sys; sys.modules['cv2'] = None; import fer_2013.serving.api"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-400:]


def _opencv_boxes(
    image: Image.Image, max_side: int = 640
) -> list[tuple[float, float, float, float]]:
    """What the pre-ONNX detector did: cv2.FaceDetectorYN on the image downscaled to max_side."""
    import cv2

    scale = min(1.0, max_side / max(image.size))
    rgb = image
    if scale < 1.0:
        rgb = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
    ref = cv2.FaceDetectorYN.create(str(REFERENCE), "", (320, 320), 0.6)
    ref.setInputSize((rgb.width, rgb.height))
    _, found = ref.detect(np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1]))
    return [tuple(float(int(v / scale)) for v in f[:4]) for f in found]  # type: ignore[misc]


@pytest.mark.skipif(not REFERENCE.exists(), reason="needs the OpenCV reference model")
def test_matches_opencv_on_the_fixture_and_a_grid(detector: YuNetDetector) -> None:
    face = Image.open(FACE).convert("RGB")
    grid = Image.new(
        "RGB", (3 * face.width, 2 * face.height)
    )  # 780 px: also exercises the downscale
    for i in range(3):
        for j in range(2):
            grid.paste(face, (i * face.width, j * face.height))
    for image in (face, grid):
        want = _opencv_boxes(image)
        got = detector.detect(image)
        assert len(got) == len(want) >= 1
        for box in want:
            assert max(_iou(box, (b.x, b.y, b.w, b.h)) for b in got) > 0.99


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)
