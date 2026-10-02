#!/usr/bin/env python3
"""Generate the golden files that the browser demo is tested against.

The Python serving pipeline is the reference: every stage of the JS port in
`web/` is compared with what this script records. Each task of the port adds
its own section to `build_golden()`.

Usage:
    uv run python scripts/make_web_golden.py --fixtures   # (re)write PNG fixtures
    uv run python scripts/make_web_golden.py              # write web/tests/golden
    uv run python scripts/make_web_golden.py --check      # fail if golden is stale
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import PIL
from PIL import Image

from fer_2013.serving.api import _clamp
from fer_2013.serving.classifier import Classifier
from fer_2013.serving.detector import YuNetDetector
from fer_2013.serving.preprocess import MEAN, RESIZE_TO, STD, crop_gray, to_input

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
GOLDEN_DIR = ROOT / "web" / "tests" / "golden"
FIXTURES_DIR = ROOT / "web" / "tests" / "fixtures"
SOURCE_FACE = ROOT / "tests" / "serving" / "fixtures" / "face.jpg"
PINNED_HASHES = ROOT / "serving" / "models.sha256"
FIXTURE_NAMES = ("face.png", "two_faces.png", "large_800.png")


def write_fixtures(out_dir: Path = FIXTURES_DIR) -> None:
    """PNG (lossless) test images: the face, two faces, and an 800x800 version."""
    out_dir.mkdir(parents=True, exist_ok=True)
    face = Image.open(SOURCE_FACE).convert("RGB")
    two = Image.new("RGB", (face.width * 2, face.height))
    two.paste(face, (0, 0))
    two.paste(face, (face.width, 0))
    face.save(out_dir / "face.png")
    two.save(out_dir / "two_faces.png")
    face.resize((800, 800), Image.Resampling.BICUBIC).save(out_dir / "large_800.png")


def pinned_hashes(path: Path = PINNED_HASHES) -> dict[str, str]:
    """name -> sha256 from serving/models.sha256 (sha256sum format)."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_meta() -> dict[str, Any]:
    pins = pinned_hashes()
    classifier = Classifier(
        MODELS_DIR / "fer_resnet18.onnx", MODELS_DIR / "fer_fc_weight.npy"
    )
    return {
        "pillow": PIL.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "classifier_sha256": pins["fer_resnet18.onnx"],
        "fc_weight_sha256": pins["fer_fc_weight.npy"],
        "yunet_2023_sha256": pins["face_detection_yunet_2023mar.onnx"],
        "yunet_2026_sha256": pins["face_detection_yunet_2026may.onnx"],
        "temperature": classifier.temperature,
        "fixtures": {name: _sha256(FIXTURES_DIR / name) for name in FIXTURE_NAMES},
    }


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


_FILTERS = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def _u8_case(
    name: str, img: Image.Image, size: tuple[int, int], filt: str
) -> dict[str, Any]:
    src = np.asarray(img)
    out = np.asarray(img.resize(size, _FILTERS[filt]))
    return {
        "name": name,
        "kind": "u8",
        "filter": filt,
        "w": img.width,
        "h": img.height,
        "channels": 1 if src.ndim == 2 else src.shape[2],
        "out_w": size[0],
        "out_h": size[1],
        "input_b64": _b64(src),
        "output_b64": _b64(out),
    }


def build_resample() -> dict[str, Any]:
    """Resampling and grayscale cases the JS port must reproduce exactly (8 bit) or to 1e-4 (float)."""
    face = Image.open(FIXTURES_DIR / "face.png").convert("RGB")
    gray = face.convert("L")

    # The detector's 640 px cap: too big to store, so the fixture is the input and
    # the golden keeps only the hash of the 640x640 result.
    large = Image.open(FIXTURES_DIR / "large_800.png").convert("RGB")
    capped = np.asarray(large.resize((640, 640), _FILTERS["bilinear"]))
    cap_case = {
        "name": "rgb_800_to_640_bilinear",
        "kind": "u8",
        "filter": "bilinear",
        "w": 800,
        "h": 800,
        "channels": 3,
        "out_w": 640,
        "out_h": 640,
        "input_fixture": "large_800.png",
        "output_sha256": hashlib.sha256(capped.tobytes()).hexdigest(),
    }

    face48 = np.asarray(
        gray.crop((50, 40, 185, 175)).resize((48, 48), _FILTERS["lanczos"])
    )
    plane = face48.astype(np.float32) / np.float32(255.0)
    big = np.asarray(
        Image.fromarray(plane, "F").resize((224, 224), _FILTERS["bicubic"]),
        dtype=np.float32,
    )
    float_case = {
        "name": "f_48_to_224_bicubic",
        "kind": "f32",
        "filter": "bicubic",
        "w": 48,
        "h": 48,
        "channels": 1,
        "out_w": 224,
        "out_h": 224,
        "input_b64": _b64(plane),
        "output_b64": _b64(big),
    }

    alpha = (
        (np.arange(face.width * face.height) % 256)
        .astype(np.uint8)
        .reshape(face.height, face.width)
    )
    rgba = np.dstack([np.asarray(face), alpha])
    gray_cases = [
        {
            "name": f"rgb_to_gray_stride{stride}",
            "kind": "gray",
            "w": face.width,
            "h": face.height,
            "stride": stride,
            "input_b64": _b64(np.asarray(face) if stride == 3 else rgba),
            "output_b64": _b64(np.asarray(gray)),
        }
        for stride in (3, 4)
    ]

    cases = [
        cap_case,
        _u8_case(
            "l_135_to_48_lanczos", gray.crop((50, 40, 185, 175)), (48, 48), "lanczos"
        ),
        _u8_case(
            "l_90x113_to_48_lanczos", gray.crop((60, 30, 150, 143)), (48, 48), "lanczos"
        ),
        _u8_case(
            "l_24_to_48_lanczos", gray.crop((100, 100, 124, 124)), (48, 48), "lanczos"
        ),
        _u8_case("rgb_260_to_130_bilinear", face, (130, 130), "bilinear"),
        _u8_case(
            "l_90x113_to_30x50_bicubic",
            gray.crop((60, 30, 150, 143)),
            (30, 50),
            "bicubic",
        ),
        float_case,
        *gray_cases,
    ]
    return {"cases": cases}


def cv_detect_floats(image: Image.Image) -> list[dict[str, float]]:
    """What `YuNetDetector` computes, but with float boxes (it truncates them to int).

    Same rules: longest side capped at 640 with a bilinear resize, OpenCV
    `FaceDetectorYN` (2023mar model, score 0.6, NMS 0.3), boxes mapped back to
    the original size, sorted by score, at most 10.
    """
    rgb = image.convert("RGB")
    scale = min(1.0, 640 / max(rgb.size))
    if scale < 1.0:
        size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
        rgb = rgb.resize(size, Image.Resampling.BILINEAR)
    bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
    det = cv2.FaceDetectorYN.create(
        str(MODELS_DIR / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6
    )
    det.setInputSize((rgb.width, rgb.height))
    _, faces = det.detect(bgr)
    if faces is None:
        return []
    boxes = [
        {
            "x": float(f[0] / scale),
            "y": float(f[1] / scale),
            "w": float(f[2] / scale),
            "h": float(f[3] / scale),
            "score": float(f[-1]),
        }
        for f in faces
    ]
    boxes.sort(key=lambda b: b["score"], reverse=True)
    return boxes[:10]


def build_detector() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in FIXTURE_NAMES:
        img = Image.open(FIXTURES_DIR / name).convert("RGB")
        out[name] = {"w": img.width, "h": img.height, "faces": cv_detect_floats(img)}
    return {"fixtures": out}


def build_preprocess() -> dict[str, Any]:
    """Per detected face: the clamped box, the gray crop, the 48x48 face and the 224x224 plane."""
    detector = YuNetDetector(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
    out: dict[str, Any] = {}
    for name in FIXTURE_NAMES:
        img = Image.open(FIXTURES_DIR / name).convert("RGB")
        faces = []
        for raw in detector.detect(img):
            box = _clamp(raw, img.width, img.height)
            gray = crop_gray(img, box) if box is not None else None
            if box is None or gray is None:
                continue
            face48 = (
                gray
                if gray.size == (48, 48)
                else gray.resize((48, 48), _FILTERS["lanczos"])
            )
            x = np.asarray(face48, dtype=np.float32) / np.float32(255.0)
            plane = np.asarray(
                Image.fromarray(x, "F").resize(RESIZE_TO, _FILTERS["bicubic"]),
                dtype=np.float32,
            )
            tensor = to_input(gray)[0]
            reference = (
                plane[None] - np.asarray(MEAN, np.float32)[:, None, None]
            ) / np.asarray(STD, np.float32)[:, None, None]
            assert np.allclose(tensor, reference, atol=1e-6), (
                "plane/normalization out of sync"
            )
            faces.append(
                {
                    "box": {"x": box.x, "y": box.y, "w": box.w, "h": box.h},
                    "crop_w": gray.width,
                    "crop_h": gray.height,
                    "crop_b64": _b64(np.asarray(gray)),
                    "face48_b64": _b64(np.asarray(face48)),
                    "plane224_b64": _b64(plane),
                }
            )
        out[name] = {"w": img.width, "h": img.height, "faces": faces}
    return {"fixtures": out}


def build_golden() -> dict[str, Any]:
    """Section name -> JSON-serializable content. Binary data goes in `*_b64` fields."""
    return {
        "meta": build_meta(),
        "resample": build_resample(),
        "detector": build_detector(),
        "preprocess": build_preprocess(),
    }


def write_golden(out_dir: Path = GOLDEN_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in build_golden().items():
        (out_dir / f"{name}.json").write_text(
            json.dumps(content, sort_keys=True) + "\n"
        )


def _diff(a: Any, b: Any, tol: float, path: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}/{key}: present on one side only")
            else:
                _diff(a[key], b[key], tol, f"{path}/{key}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _diff(x, y, tol, f"{path}[{i}]", out)
    elif (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    ):
        if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
            out.append(f"{path}: {a} != {b}")
    elif a != b:
        out.append(f"{path}: {str(a)[:60]} != {str(b)[:60]}")


def check_golden(golden_dir: Path = GOLDEN_DIR, tol: float = 1e-6) -> list[str]:
    """Differences between the committed golden files and a fresh build."""
    problems: list[str] = []
    fresh = json.loads(json.dumps(build_golden()))  # same float round-trip as the files
    for name, content in fresh.items():
        file = golden_dir / f"{name}.json"
        if not file.exists():
            problems.append(f"{name}.json is missing")
            continue
        _diff(json.loads(file.read_text()), content, tol, name, problems)
    return problems[:20]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or check the web golden files."
    )
    parser.add_argument("--out", type=Path, default=GOLDEN_DIR)
    parser.add_argument(
        "--check", action="store_true", help="fail if the files are stale"
    )
    parser.add_argument(
        "--fixtures", action="store_true", help="rewrite the PNG fixtures"
    )
    args = parser.parse_args(argv)

    if args.fixtures:
        write_fixtures()
        print(f"wrote fixtures to {FIXTURES_DIR}")
    if args.check:
        problems = check_golden(args.out)
        for problem in problems:
            print("STALE:", problem)
        return 1 if problems else 0
    write_golden(args.out)
    print(f"wrote golden files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
