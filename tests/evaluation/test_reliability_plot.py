from pathlib import Path

import numpy as np

from fer_2013.evaluation.plots import plot_reliability_diagram


def test_reliability_diagram_is_written(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    conf = np.clip(rng.random(15), 0.1, 1.0)
    bins = (conf, conf * 0.9, rng.integers(1, 50, 15))
    out = tmp_path / "reliability.png"
    plot_reliability_diagram({"raw": bins, "calibrated": bins}, out)
    assert out.stat().st_size > 1000
