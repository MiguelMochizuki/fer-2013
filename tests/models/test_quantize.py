from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from fer_2013.models.quantize import (
    QuantizationError,
    quantize_classifier,
    verify_quantized,
)

FP32 = Path(
    "models/fer_resnet18_fp32.onnx"
)  # kept next to the int8 model by scripts/quantize_onnx.py
DATA = Path("data/processed")

pytestmark = pytest.mark.skipif(
    not (FP32.exists() and (DATA / "X_train.npy").exists()),
    reason="needs the exported fp32 model and data/processed",
)


@pytest.fixture(scope="module")
def int8_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("q") / "fer_resnet18.onnx"
    images = np.load(DATA / "X_train.npy")
    quantize_classifier(FP32, out, images, n_calibration=60)
    return out


def test_quantized_model_keeps_the_contract_and_is_much_smaller(
    int8_path: Path,
) -> None:
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == ["logits", "features", "probs"]
    assert sess.get_modelmeta().custom_metadata_map["temperature"]
    assert int8_path.stat().st_size < FP32.stat().st_size / 3
    logits, features, probs = sess.run(
        None, {"input": np.zeros((2, 3, 224, 224), np.float32)}
    )
    assert logits.shape == (2, 7) and features.shape == (2, 512, 7, 7)
    # probs are a float softmax of the logits / T, not snapped to int8 steps
    t = float(sess.get_modelmeta().custom_metadata_map["temperature"])
    z = logits / t
    want = np.exp(z - z.max(axis=1, keepdims=True))
    want /= want.sum(axis=1, keepdims=True)
    assert np.allclose(probs, want, atol=1e-5)


def test_verify_passes_on_a_good_model_and_fails_on_an_impossible_bar(
    int8_path: Path,
) -> None:
    images = np.load(DATA / "X_val.npy")[:150]
    labels = np.load(DATA / "y_val.npy")[:150]
    report = verify_quantized(
        FP32, int8_path, images, labels, min_agreement=0.8, max_accuracy_drop=0.1
    )
    assert 0.8 <= report["agreement"] <= 1.0
    with pytest.raises(QuantizationError):
        verify_quantized(FP32, int8_path, images, labels, min_agreement=1.01)
