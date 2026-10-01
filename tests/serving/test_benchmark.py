import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).parents[2] / "scripts" / "benchmark.py"
)
assert _spec is not None and _spec.loader is not None
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def test_percentiles_known_values() -> None:
    p = benchmark.percentiles([float(i) for i in range(1, 101)])
    assert p["p50"] == pytest.approx(50.5)
    assert p["p95"] == pytest.approx(95.05)


def test_percentiles_single_sample() -> None:
    assert benchmark.percentiles([7.0]) == {"p50": 7.0, "p95": 7.0}


def test_percentiles_unsorted_input() -> None:
    assert benchmark.percentiles([3.0, 1.0, 2.0])["p50"] == 2.0
