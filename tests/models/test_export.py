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
    assert [o.name for o in sess.get_outputs()] == ["logits", "features"]
    x = np.zeros((2, 3, 224, 224), dtype=np.float32)
    logits, features = sess.run(None, {"input": x})
    assert logits.shape == (2, 7)
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
