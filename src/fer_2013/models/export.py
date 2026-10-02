"""Export the trained ResNet18 to ONNX for the torch-free serving image.

The exported graph has three outputs: the raw class logits, the layer4
activations (N, 512, 7, 7), from which serving computes Grad-CAM++ in numpy,
and `probs`, the softmax of the logits divided by a calibration temperature.
Grad-CAM++ must keep using the raw logits and weights: its closed form is not
invariant to their scale. The Linear head weights are saved next to the model
as a .npy file, and the temperature is stored in the ONNX metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torchvision.models import ResNet

ONNX_NAME = "fer_resnet18.onnx"
FC_WEIGHT_NAME = "fer_fc_weight.npy"
OPSET = 17


class _WithFeatures(nn.Module):
    """ResNet forward that also returns the layer4 feature maps."""

    def __init__(self, model: ResNet, temperature: float) -> None:
        super().__init__()
        self.m = model
        self.temperature = temperature

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        feats = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        logits = m.fc(torch.flatten(m.avgpool(feats), 1))
        return logits, feats, torch.softmax(logits / self.temperature, dim=1)


def export_classifier(
    model: ResNet, out_dir: Path, temperature: float = 1.0
) -> tuple[Path, Path]:
    """Write fer_resnet18.onnx and fer_fc_weight.npy into `out_dir`.

    Args:
        model: trained ResNet18.
        out_dir: output directory.
        temperature: calibration temperature for the `probs` output (1.0 means
            a plain softmax).

    Raises:
        ValueError: `temperature` is not positive.
        AssertionError: onnxruntime outputs differ from PyTorch's.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / ONNX_NAME
    fc_path = out_dir / FC_WEIGHT_NAME
    tmp_path = out_dir / (ONNX_NAME + ".tmp")

    model.eval()
    wrapper = _WithFeatures(model, temperature).eval()
    dummy = torch.randn(2, 3, 224, 224)
    batch = {0: "batch"}
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(tmp_path),
        input_names=["input"],
        output_names=["logits", "features", "probs"],
        dynamic_axes={
            "input": batch,
            "logits": batch,
            "features": batch,
            "probs": batch,
        },
        opset_version=OPSET,
        dynamo=False,
    )

    try:
        exported = onnx.load(str(tmp_path))
        entry = exported.metadata_props.add()
        entry.key, entry.value = "temperature", str(temperature)
        onnx.save(exported, str(tmp_path))

        with torch.no_grad():
            want_logits, want_feats, want_probs = (t.numpy() for t in wrapper(dummy))
        sess = ort.InferenceSession(str(tmp_path), providers=["CPUExecutionProvider"])
        got_logits, got_feats, got_probs = sess.run(None, {"input": dummy.numpy()})
        np.testing.assert_allclose(got_logits, want_logits, atol=1e-4)
        np.testing.assert_allclose(got_feats, want_feats, atol=1e-4)
        np.testing.assert_allclose(got_probs, want_probs, atol=1e-5)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(onnx_path)  # only a verified model gets the final name

    fc_weight = model.fc[1].weight.detach().cpu().numpy().astype(np.float32)
    np.save(fc_path, fc_weight)
    return onnx_path, fc_path
