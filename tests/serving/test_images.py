import io
import warnings

import pytest
from PIL import Image

from fer_2013.serving.images import (
    MAX_UPLOAD_BYTES,
    ImageRejected,
    decode_image,
    read_limited,
)


def _encode(img: Image.Image, fmt: str, **kw: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def test_read_limited_accepts_exactly_limit() -> None:
    half = MAX_UPLOAD_BYTES // 2
    data = read_limited([b"a" * half, b"b" * (MAX_UPLOAD_BYTES - half)])
    assert len(data) == MAX_UPLOAD_BYTES


def test_read_limited_rejects_limit_plus_one() -> None:
    with pytest.raises(ImageRejected) as e:
        read_limited(iter([b"a" * MAX_UPLOAD_BYTES, b"b"]))
    assert e.value.status_code == 413


def test_decode_rejects_non_image() -> None:
    with pytest.raises(ImageRejected) as e:
        decode_image(b"not an image")
    assert e.value.status_code == 422


def test_decode_rejects_truncated_jpeg() -> None:
    jpeg = _encode(Image.effect_noise((200, 200), 80).convert("RGB"), "JPEG")
    with pytest.raises(ImageRejected) as e:
        decode_image(jpeg[: len(jpeg) // 2])
    assert e.value.status_code == 422


def test_decode_rejects_gif() -> None:
    with pytest.raises(ImageRejected) as e:
        decode_image(_encode(Image.new("P", (10, 10)), "GIF"))
    assert e.value.status_code == 422


def test_decode_rejects_decompression_bomb() -> None:
    bomb = _encode(Image.new("1", (6000, 6000)), "PNG")  # 36 MP, tiny on disk
    assert len(bomb) < MAX_UPLOAD_BYTES
    with pytest.raises(ImageRejected) as e:
        decode_image(bomb)
    assert e.value.status_code == 422


def test_decode_applies_exif_rotation() -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 clockwise to display
    jpeg = _encode(Image.new("RGB", (40, 20)), "JPEG", exif=exif.tobytes())
    assert decode_image(jpeg).size == (20, 40)


@pytest.mark.parametrize("mode", ["RGBA", "P", "L", "I;16", "LA"])
def test_decode_accepts_other_png_modes(mode: str) -> None:
    img = decode_image(_encode(Image.new(mode, (32, 32)), "PNG"))
    assert img.mode == "RGB"
    assert img.size == (32, 32)


def test_pixel_limit_does_not_depend_on_warning_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """warnings.catch_warnings is process-wide, so another thread can undo it."""
    bomb = _encode(Image.new("1", (6000, 6000)), "PNG")  # 36 MP > 25 MP limit
    monkeypatch.setattr(warnings, "simplefilter", lambda *a, **k: None)
    with pytest.raises(ImageRejected) as e:
        decode_image(bomb)
    assert e.value.status_code == 422
