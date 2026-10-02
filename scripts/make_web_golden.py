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

from fer_2013.serving.classifier import Classifier

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
        "temperature": classifier.temperature,
        "fixtures": {name: _sha256(FIXTURES_DIR / name) for name in FIXTURE_NAMES},
    }


def build_golden() -> dict[str, Any]:
    """Section name -> JSON-serializable content. Binary data goes in `*_b64` fields."""
    return {"meta": build_meta()}


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
