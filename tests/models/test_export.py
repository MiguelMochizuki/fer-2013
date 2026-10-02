from pathlib import Path

from torchvision.models import ResNet

from fer_2013.models import export

import numpy as np
import pytest
import onnxruntime as ort


def test_export_outputs_logits_and_features(onnx_models: tuple[Path, Path]) -> None:
    onnx_path, fc_path = onnx_models
    assert onnx_path.name == "fer_resnet18.onnx"
    assert fc_path.name == "fer_fc_weight.npy"
    assert np.load(fc_path).shape == (7, 512)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == ["logits", "features", "probs"]
    x = np.zeros((2, 3, 224, 224), dtype=np.float32)
    logits, features, probs = sess.run(None, {"input": x})
    assert logits.shape == (2, 7)
    assert probs.shape == (2, 7)
    assert features.shape == (2, 512, 7, 7)


def test_failed_parity_leaves_no_onnx_behind(
    torch_model: ResNet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("parity")

    monkeypatch.setattr(np.testing, "assert_allclose", boom)
    with pytest.raises(AssertionError):
        export.export_classifier(torch_model, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_export_adds_a_calibrated_probs_output(torch_model: ResNet, tmp_path: Path) -> None:
    onnx_path, _ = export.export_classifier(torch_model, tmp_path, temperature=2.5)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == ["logits", "features", "probs"]
    x = np.random.default_rng(0).normal(size=(2, 3, 224, 224)).astype(np.float32)
    logits, _, probs = sess.run(None, {"input": x})
    z = logits / 2.5
    expected = np.exp(z - z.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(probs, expected, atol=1e-5)
    assert sess.get_modelmeta().custom_metadata_map["temperature"] == "2.5"


def test_default_temperature_gives_plain_softmax(torch_model: ResNet, tmp_path: Path) -> None:
    onnx_path, _ = export.export_classifier(torch_model, tmp_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    logits, _, probs = sess.run(None, {"input": x})
    assert np.allclose(probs.sum(), 1.0)
    assert probs.argmax() == logits.argmax()


def test_non_positive_temperature_is_rejected(torch_model: ResNet, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        export.export_classifier(torch_model, tmp_path, temperature=0.0)
