from pathlib import Path

import numpy as np
import torch
from torch import nn

from fer_2013.serving.classifier import Classifier


def test_classifier_matches_torch(
    onnx_models: tuple[Path, Path], torch_model: nn.Module
) -> None:
    clf = Classifier(*onnx_models)
    x = np.random.default_rng(0).normal(size=(3, 3, 224, 224)).astype(np.float32)
    probs, features = clf.predict(x)
    with torch.no_grad():
        expected = torch.softmax(torch_model(torch.from_numpy(x)), dim=1).numpy()
    assert probs.shape == (3, 7)
    assert features.shape == (3, 512, 7, 7)
    assert np.allclose(probs, expected, atol=1e-4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_predict_batch_dim_is_dynamic(onnx_models: tuple[Path, Path]) -> None:
    clf = Classifier(*onnx_models)
    for n in (1, 3):
        probs, _ = clf.predict(np.zeros((n, 3, 224, 224), dtype=np.float32))
        assert probs.shape == (n, 7)


def test_fc_weight_and_sha(onnx_models: tuple[Path, Path]) -> None:
    clf = Classifier(*onnx_models)
    assert clf.fc_weight.shape == (7, 512)
    assert len(clf.sha) == 8
    int(clf.sha, 16)
