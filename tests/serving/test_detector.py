from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from fer_2013.serving.detector import Box, YuNetDetector

MODEL = Path("models/face_detection_yunet_2023mar.onnx")
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
