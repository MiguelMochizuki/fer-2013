import base64
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from fer_2013.evaluation.evaluate import EMOTION_LABELS, load_checkpoint
from fer_2013.evaluation.gradcam import generate_heatmaps
from fer_2013.models.cnn import build_resnet18
from fer_2013.serving.classifier import Classifier
from fer_2013.serving.gradcam import gradcam_pp, overlay_png_b64

CHECKPOINT = Path("checkpoints/ferplus/best.pt")


def test_zero_features_give_zero_map_no_nan() -> None:
    cam = gradcam_pp(
        np.zeros((512, 7, 7), np.float32), np.ones((7, 512), np.float32), 3
    )
    assert cam.shape == (7, 7)
    assert not np.isnan(cam).any()
    assert cam.max() == 0.0


def test_map_is_in_unit_range() -> None:
    rng = np.random.default_rng(0)
    feats = np.abs(rng.normal(size=(512, 7, 7))).astype(np.float32)
    fc = rng.normal(size=(7, 512)).astype(np.float32)
    cam = gradcam_pp(feats, fc, 2)
    assert cam.dtype == np.float32
    assert cam.min() >= 0.0 and cam.max() <= 1.0


def test_overlay_returns_decodable_png_128() -> None:
    gray = Image.new("L", (60, 60), 128)
    cam = np.random.default_rng(0).random((7, 7)).astype(np.float32)
    png = base64.b64decode(overlay_png_b64(gray, cam))
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG"
    assert img.size == (128, 128)


def _assert_parity(model: nn.Module, clf: Classifier) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 3, 224, 224)).astype(np.float32)
    _, features = clf.predict(x)
    for c in range(len(EMOTION_LABELS)):  # every class, not just the argmax
        ref = generate_heatmaps(model, torch.from_numpy(x), [c] * len(x))
        for i in range(len(x)):
            mine = gradcam_pp(features[i], clf.fc_weight, c)
            up = cv2.resize(mine, (224, 224), interpolation=cv2.INTER_LINEAR)
            assert np.corrcoef(up.ravel(), ref[i].ravel())[0, 1] > 0.999


def test_parity_with_pytorch_grad_cam(
    torch_model: nn.Module, onnx_models: tuple[Path, Path]
) -> None:
    _assert_parity(torch_model, Classifier(*onnx_models))


@pytest.mark.slow
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs checkpoints/ferplus/best.pt")
def test_parity_on_trained_checkpoint(
    tmp_path: Path,
) -> None:
    from fer_2013.models.export import export_classifier

    model = build_resnet18(num_classes=len(EMOTION_LABELS), pretrained=False)
    load_checkpoint(CHECKPOINT, model)
    _assert_parity(model, Classifier(*export_classifier(model, tmp_path)))
