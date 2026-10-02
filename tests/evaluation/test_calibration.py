import numpy as np
import pytest

from fer_2013.evaluation.calibration import (
    brier_score,
    calibrated_probs,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    reliability_bins,
)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    out: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return out


def _synthetic(
    t_true: float, n: int = 20000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Logits whose labels were drawn from softmax(logits / t_true)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 3, size=(n, 7))
    p = _softmax(logits / t_true)
    u = rng.random(n)[:, None]
    labels = np.minimum((p.cumsum(axis=1) < u).sum(axis=1), 6)
    return logits, labels


def test_ece_is_zero_when_confidence_matches_accuracy() -> None:
    probs = np.tile([0.75, 0.25], (100, 1))
    targets = np.array([0] * 75 + [1] * 25)  # 75% right at 75% confidence
    assert expected_calibration_error(probs, targets) == pytest.approx(0.0, abs=1e-9)


def test_ece_of_a_fully_confident_coin_flip_is_one_half() -> None:
    probs = np.tile([1.0, 0.0], (100, 1))
    targets = np.array([0] * 50 + [1] * 50)  # confidence 1.0, accuracy 0.5
    assert expected_calibration_error(probs, targets) == pytest.approx(0.5)


def test_reliability_bins_partition_the_samples() -> None:
    probs = np.tile([0.75, 0.25], (100, 1))
    targets = np.array([0] * 75 + [1] * 25)
    conf, acc, count = reliability_bins(probs, targets, n_bins=15)
    assert count.sum() == 100
    (filled,) = np.nonzero(count)
    assert len(filled) == 1
    assert conf[filled[0]] == pytest.approx(0.75)
    assert acc[filled[0]] == pytest.approx(0.75)
    assert np.isnan(conf[count == 0]).all()


def test_fit_temperature_recovers_the_known_value() -> None:
    logits, labels = _synthetic(t_true=2.4)
    assert fit_temperature(logits, labels) == pytest.approx(2.4, abs=0.15)


def test_fit_temperature_is_one_for_calibrated_logits() -> None:
    logits, labels = _synthetic(t_true=1.0)
    assert fit_temperature(logits, labels) == pytest.approx(1.0, abs=0.08)


def test_calibration_lowers_nll_and_ece_without_changing_predictions() -> None:
    logits, labels = _synthetic(t_true=2.4)
    t = fit_temperature(logits, labels)
    before, after = _softmax(logits), calibrated_probs(logits, t)
    assert negative_log_likelihood(logits, labels, t) < negative_log_likelihood(
        logits, labels
    )
    assert expected_calibration_error(after, labels) < expected_calibration_error(
        before, labels
    )
    assert (after.argmax(axis=1) == logits.argmax(axis=1)).all()
    assert np.allclose(after.sum(axis=1), 1.0)


def test_calibrated_probs_are_stable_for_extreme_logits() -> None:
    probs = calibrated_probs(np.array([[1000.0, -1000.0, 0.0]]), 1.0)
    assert np.isfinite(probs).all()
    assert probs[0, 0] == pytest.approx(1.0)


def test_brier_score_known_values() -> None:
    certain_right = np.array([[1.0, 0.0], [0.0, 1.0]])
    certain_wrong = np.array([[0.0, 1.0], [1.0, 0.0]])
    uniform = np.full((2, 2), 0.5)
    targets = np.array([0, 1])
    assert brier_score(certain_right, targets) == pytest.approx(0.0)
    assert brier_score(certain_wrong, targets) == pytest.approx(2.0)
    assert brier_score(uniform, targets) == pytest.approx(0.5)
