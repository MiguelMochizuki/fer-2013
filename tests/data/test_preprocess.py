"""Tests for fer_2013.data.preprocess."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fer_2013.data import preprocess as pp


def _pixels(value: int) -> str:
    """A valid 48*48 pixel string filled with the given value."""
    return " ".join([str(value)] * (48 * 48))


def _make_csv(tmp_path: Path, rows: list[tuple[int, str, str]]) -> Path:
    """Build a small fer2013-like CSV at tmp_path/fer2013.csv."""
    path = tmp_path / "fer2013.csv"
    pd.DataFrame(rows, columns=["emotion", "pixels", "Usage"]).to_csv(path, index=False)
    return path


# ==============================
# _parse_pixels
# ==============================


def test_parse_pixels_returns_correct_shape_and_dtype() -> None:
    arr = pp._parse_pixels(_pixels(7))
    assert arr.shape == (48, 48)
    assert arr.dtype == np.uint8
    assert (arr == 7).all()


def test_parse_pixels_raises_on_wrong_count() -> None:
    with pytest.raises(pp.MalformedRowError):
        pp._parse_pixels("1 2 3")


# ==============================
# preprocess_fer2013
# ==============================


def test_preprocess_writes_six_npy_files(tmp_path: Path) -> None:
    csv = _make_csv(
        tmp_path,
        [
            (0, _pixels(1), "Training"),
            (1, _pixels(2), "Training"),
            (2, _pixels(3), "PublicTest"),
            (3, _pixels(4), "PrivateTest"),
        ],
    )
    out = tmp_path / "processed"
    written = pp.preprocess_fer2013(csv, out)

    for name in (
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
    ):
        assert name in written
        assert (out / name).exists()


def test_preprocess_raises_if_csv_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pp.preprocess_fer2013(tmp_path / "nope.csv", tmp_path / "out")


def test_preprocess_raises_on_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "fer2013.csv"
    pd.DataFrame({"emotion": [0], "pixels": [_pixels(1)]}).to_csv(path, index=False)
    with pytest.raises(pp.PreprocessError):
        pp.preprocess_fer2013(path, tmp_path / "out")


def test_preprocess_raises_on_unknown_usage(tmp_path: Path) -> None:
    csv = _make_csv(tmp_path, [(0, _pixels(1), "Weird")])
    with pytest.raises(pp.PreprocessError):
        pp.preprocess_fer2013(csv, tmp_path / "out")


def test_preprocess_arrays_have_expected_shape_and_dtype(tmp_path: Path) -> None:
    csv = _make_csv(
        tmp_path,
        [
            (0, _pixels(1), "Training"),
            (1, _pixels(2), "Training"),
            (2, _pixels(3), "PublicTest"),
        ],
    )
    out = tmp_path / "processed"
    pp.preprocess_fer2013(csv, out)

    X_train = np.load(out / "X_train.npy")
    y_train = np.load(out / "y_train.npy")
    X_val = np.load(out / "X_val.npy")
    y_val = np.load(out / "y_val.npy")

    assert X_train.shape == (2, 48, 48)
    assert X_train.dtype == np.uint8
    assert y_train.shape == (2,)
    assert y_train.dtype == np.int64
    assert X_val.shape == (1, 48, 48)
    assert y_val.shape == (1,)
    assert y_val.tolist() == [2]


def test_preprocess_splits_by_usage(tmp_path: Path) -> None:
    csv = _make_csv(
        tmp_path,
        [
            (0, _pixels(1), "Training"),
            (1, _pixels(2), "PublicTest"),
            (2, _pixels(3), "PrivateTest"),
        ],
    )
    out = tmp_path / "processed"
    pp.preprocess_fer2013(csv, out)

    assert np.load(out / "y_train.npy").tolist() == [0]
    assert np.load(out / "y_val.npy").tolist() == [1]
    assert np.load(out / "y_test.npy").tolist() == [2]


def test_preprocess_is_idempotent(tmp_path: Path) -> None:
    """Second call must not rewrite the .npy files."""
    csv = _make_csv(tmp_path, [(0, _pixels(1), "Training")])
    out = tmp_path / "processed"
    pp.preprocess_fer2013(csv, out)

    before = (out / "X_train.npy").stat().st_mtime_ns
    pp.preprocess_fer2013(csv, out)
    after = (out / "X_train.npy").stat().st_mtime_ns

    assert before == after


def test_preprocess_force_rewrites(tmp_path: Path) -> None:
    csv = _make_csv(tmp_path, [(0, _pixels(1), "Training")])
    out = tmp_path / "processed"
    pp.preprocess_fer2013(csv, out)

    before = (out / "X_train.npy").stat().st_mtime_ns
    pp.preprocess_fer2013(csv, out, force=True)
    after = (out / "X_train.npy").stat().st_mtime_ns

    assert after > before


# ==============================
# constants
# ==============================


def test_constants() -> None:
    assert pp.IMAGE_SHAPE == (48, 48)
    assert pp.PIXEL_COUNT == 48 * 48
    assert pp.SPLIT_MAP == {
        "Training": "train",
        "PublicTest": "val",
        "PrivateTest": "test",
    }
    assert pp.EMOTION_LABELS == (
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    )


# ==============================
# preprocess_ferplus
# ==============================

_VOTE_COLS = [*pp.FERPLUS_COLUMNS, "contempt", "unknown", "NF"]


def _make_ferplus_csv(tmp_path: Path, rows: list[tuple[str, list[int]]]) -> Path:
    path = tmp_path / "fer2013new.csv"
    df = pd.DataFrame([v for _, v in rows], columns=_VOTE_COLS)
    df.insert(0, "Usage", [u for u, _ in rows])
    df.to_csv(path, index=False)
    return path


def test_ferplus_soft_labels_follow_votes_and_drop_non_emotion_rows(
    tmp_path: Path,
) -> None:
    csv = _make_csv(
        tmp_path,
        [
            (0, _pixels(1), "Training"),
            (0, _pixels(2), "Training"),
            (0, _pixels(3), "Training"),
            (0, _pixels(4), "PublicTest"),
            (0, _pixels(5), "PrivateTest"),
        ],
    )
    plus = _make_ferplus_csv(
        tmp_path,
        [
            # anger fer order: anger disgust fear happy sad surprise neutral | contempt unknown NF
            ("Training", [0, 0, 0, 6, 0, 0, 4, 0, 0, 0]),  # happy 0.6 / neutral 0.4
            ("Training", [0, 0, 0, 0, 0, 0, 2, 0, 8, 0]),  # unknown wins -> dropped
            ("Training", [0, 0, 0, 0, 0, 0, 3, 7, 0, 0]),  # contempt wins -> dropped
            ("PublicTest", [9, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            ("PrivateTest", [0, 0, 0, 0, 0, 10, 0, 0, 0, 0]),
        ],
    )
    out = tmp_path / "plus"
    pp.preprocess_ferplus(csv, plus, out)

    soft = np.load(out / "y_train_soft.npy")
    assert soft.shape == (1, 7) and soft.dtype == np.float32
    np.testing.assert_allclose(soft[0], [0, 0, 0, 0.6, 0, 0, 0.4])
    assert np.load(out / "y_train.npy").tolist() == [3]
    assert np.load(out / "X_train.npy")[0, 0, 0] == 1
    assert np.load(out / "y_val.npy").tolist() == [0]
    assert np.load(out / "y_test.npy").tolist() == [5]


def test_ferplus_rejects_misaligned_csvs(tmp_path: Path) -> None:
    csv = _make_csv(tmp_path, [(0, _pixels(1), "Training")])
    plus = _make_ferplus_csv(
        tmp_path, [("PublicTest", [10, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
    )
    with pytest.raises(pp.PreprocessError, match="aligned"):
        pp.preprocess_ferplus(csv, plus, tmp_path / "plus")
