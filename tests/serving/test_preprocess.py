import numpy as np
import pytest
import torch
from PIL import Image

from fer_2013.data import datamodule
from fer_2013.serving import preprocess
from fer_2013.serving.detector import Box
from fer_2013.serving.preprocess import crop_gray, to_input


def test_constants_match_datamodule() -> None:
    assert preprocess.RESIZE_TO == datamodule.RESIZE_TO
    assert list(preprocess.MEAN) == datamodule.IMAGENET_MEAN
    assert list(preprocess.STD) == datamodule.IMAGENET_STD


def test_to_input_matches_build_transforms() -> None:
    rng = np.random.default_rng(0)
    tf = datamodule.build_transforms("test")
    for _ in range(20):
        a = rng.integers(0, 256, size=(48, 48), dtype=np.uint8)
        expected = tf(torch.from_numpy(a).float().unsqueeze(0) / 255.0).numpy()
        got = to_input(Image.fromarray(a, "L"))
        assert got.shape == (1, 3, 224, 224)
        assert got.dtype == np.float32
        assert np.allclose(got[0], expected, atol=1e-3)


def test_crop_clamps_to_image() -> None:
    img = Image.new("RGB", (100, 80), (200, 100, 50))
    for box in (Box(-10, -10, 30, 30, 0.9), Box(90, 70, 40, 40, 0.9)):
        out = crop_gray(img, box)
        assert out is not None
        assert out.mode == "L"
        assert 0 < out.width <= 100 and 0 < out.height <= 80


def test_crop_empty_box_is_none() -> None:
    img = Image.new("RGB", (100, 80))
    assert crop_gray(img, Box(5, 5, 0, 10, 0.9)) is None
    assert crop_gray(img, Box(500, 500, 10, 10, 0.9)) is None


@pytest.mark.parametrize("mode", ["RGBA", "P", "I;16", "L", "CMYK"])
def test_crop_accepts_other_modes(mode: str) -> None:
    img = Image.new(mode, (64, 64))
    out = crop_gray(img, Box(8, 8, 40, 40, 0.9))
    assert out is not None
    assert out.mode == "L"


def test_crop_is_square_for_a_tall_box() -> None:
    img = Image.new("RGB", (300, 300))
    out = crop_gray(img, Box(100, 80, 90, 113, 0.9))
    assert out is not None
    assert out.width == out.height == int(113 * 1.2)


def test_crop_stays_square_next_to_the_border() -> None:
    img = Image.new("RGB", (300, 300))
    out = crop_gray(img, Box(0, 50, 60, 80, 0.9))
    assert out is not None
    assert out.width == out.height


def test_crop_of_a_box_outside_the_image_is_none() -> None:
    img = Image.new("RGB", (100, 100))
    assert crop_gray(img, Box(150, 10, 20, 20, 0.9)) is None
    assert crop_gray(img, Box(-30, 10, 20, 20, 0.9)) is None
