"""Upload reading and image decoding with hard limits."""

from __future__ import annotations

import io
import warnings
from collections.abc import Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

from fer_2013.serving.preprocess import to_gray

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 25_000_000
FORMATS = ["JPEG", "PNG", "WEBP"]

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageRejected(Exception):
    """The upload cannot be processed; `status_code` is the HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def read_limited(chunks: Iterable[bytes]) -> bytes:
    """Concatenate chunks, stopping with 413 as soon as the limit is exceeded."""
    buf = bytearray()
    for chunk in chunks:
        buf += chunk
        if len(buf) > MAX_UPLOAD_BYTES:
            raise ImageRejected(413, "file too large")
    return bytes(buf)


def decode_image(data: bytes) -> Image.Image:
    """Decode a JPEG/PNG/WebP upload to an upright RGB image."""
    try:
        with warnings.catch_warnings():
            # Pillow warns (not raises) between 1x and 2x MAX_IMAGE_PIXELS.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data), formats=FORMATS)
            img.load()  # decode fully so truncated files fail here
        img = ImageOps.exif_transpose(img)
        if img.mode.startswith("I") or img.mode == "F":
            img = to_gray(img)  # high-bit-depth: scale to 8 bits first
        return img.convert("RGB")
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
        ValueError,
    ) as e:
        raise ImageRejected(422, "invalid or unsupported image") from e
