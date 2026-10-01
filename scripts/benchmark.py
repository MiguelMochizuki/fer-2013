#!/usr/bin/env python3
"""Measure /predict latency (client-side wall time) against a running server.

Usage:
    uv run python scripts/benchmark.py --url http://localhost:7860 \\
        --image tests/serving/fixtures/face.jpg --n 50
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def percentiles(samples: list[float]) -> dict[str, float]:
    """p50 and p95 with linear interpolation between ranks."""
    s = sorted(samples)

    def pct(q: float) -> float:
        k = (len(s) - 1) * q
        lo = math.floor(k)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return {"p50": pct(0.50), "p95": pct(0.95)}


def _post(url: str, image: bytes) -> None:
    boundary = uuid.uuid4().hex
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="img.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        + image
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req) as resp:
        resp.read()


def run(url: str, image: bytes, n: int, warmup: int, explain: bool) -> list[float]:
    target = f"{url.rstrip('/')}/predict?explain={str(explain).lower()}"
    for _ in range(warmup):
        _post(target, image)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        _post(target, image)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark POST /predict latency.")
    parser.add_argument("--url", default="http://localhost:7860")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args(argv)

    image = args.image.read_bytes()
    print(f"{'explain':<8} {'n':>4} {'p50 ms':>9} {'p95 ms':>9}")
    for explain in (False, True):
        p = percentiles(run(args.url, image, args.n, args.warmup, explain))
        print(f"{explain!s:<8} {args.n:>4} {p['p50']:>9.1f} {p['p95']:>9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
