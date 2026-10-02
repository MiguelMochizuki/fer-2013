import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).parents[2] / "scripts" / "make_web_golden.py"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_web_golden", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()
needs_models = pytest.mark.skipif(
    not (gen.MODELS_DIR / "fer_resnet18.onnx").exists(),
    reason="needs the classifier model in models/",
)


@needs_models
def test_golden_files_are_fresh() -> None:
    """The committed golden files must match what the Python pipeline produces now."""
    assert gen.GOLDEN_DIR.exists(), "run: uv run python scripts/make_web_golden.py"
    assert gen.check_golden(gen.GOLDEN_DIR) == []


def test_fixtures_exist_and_golden_dir_is_inside_the_repo() -> None:
    assert gen.GOLDEN_DIR.is_relative_to(gen.ROOT)
    for name in gen.FIXTURE_NAMES:
        assert (gen.FIXTURES_DIR / name).exists(), f"missing fixture {name}"
