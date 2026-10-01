from pathlib import Path

import pytest
import torch
from torchvision.models import ResNet

from fer_2013.models.cnn import build_resnet18
from fer_2013.models.export import export_classifier


@pytest.fixture(scope="session")
def torch_model() -> ResNet:
    """Untrained ResNet18 with a fixed seed; enough to test the mechanics."""
    torch.manual_seed(0)
    model = build_resnet18(pretrained=False)
    model.eval()
    return model


@pytest.fixture(scope="session")
def onnx_models(
    torch_model: ResNet, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path]:
    return export_classifier(torch_model, tmp_path_factory.mktemp("onnx"))
