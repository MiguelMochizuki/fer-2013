"""Export the trained ResNet18 to ONNX for the torch-free serving image.

The exported graph has two outputs: the class logits and the layer4
activations (N, 512, 7, 7), from which serving computes Grad-CAM++ in numpy.
The Linear head weights are saved next to it as a .npy file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torchvision.models import ResNet

ONNX_NAME = "fer_resnet18.onnx"
FC_WEIGHT_NAME = "fer_fc_weight.npy"
OPSET = 17


class _WithFeatures(nn.Module):
    """ResNet forward that also returns the layer4 feature maps."""

    def __init__(self, model: ResNet) -> None:
        super().__init__()
        self.m = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        feats = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        logits = m.fc(torch.flatten(m.avgpool(feats), 1))
        return logits, feats


def export_classifier(model: ResNet, out_dir: Path) -> tuple[Path, Path]:
    """Write fer_resnet18.onnx and fer_fc_weight.npy into `out_dir`.

    Raises:
        AssertionError: onnxruntime outputs differ from PyTorch's.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / ONNX_NAME
    fc_path = out_dir / FC_WEIGHT_NAME

    model.eval()
    wrapper = _WithFeatures(model).eval()
    dummy = torch.randn(2, 3, 224, 224)
    batch = {0: "batch"}
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["input"],
        output_names=["logits", "features"],
        dynamic_axes={"input": batch, "logits": batch, "features": batch},
        opset_version=OPSET,
        dynamo=False,
    )

    with torch.no_grad():
        want_logits, want_feats = (t.numpy() for t in wrapper(dummy))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got_logits, got_feats = sess.run(None, {"input": dummy.numpy()})
    np.testing.assert_allclose(got_logits, want_logits, atol=1e-4)
    np.testing.assert_allclose(got_feats, want_feats, atol=1e-4)

    fc_weight = model.fc[1].weight.detach().cpu().numpy().astype(np.float32)
    np.save(fc_path, fc_weight)
    return onnx_path, fc_path
