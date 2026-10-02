"""Static int8 quantization of the exported classifier.

The fp32 graph is quantized in the QDQ format (per-channel int8 weights, uint8
activations, MinMax calibration on a random sample of training faces). Outputs
(`logits`, `features`, `probs`) and the `temperature` metadata are unchanged, so
serving and the browser load it exactly like the fp32 model. Only the quantized
graph runs faster and ships at about a quarter of the size; the softmax and the
temperature division stay in float.

`verify_quantized` is the quality gate: it compares the int8 model with the
fp32 one on held-out images and raises if predictions or accuracy drift.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

from fer_2013.serving.preprocess import to_input


class QuantizationError(RuntimeError):
    """The quantized model drifted too far from the fp32 one."""


def _inputs(images: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """(n, 48, 48) uint8 faces -> (n, 3, 224, 224) float32 model input."""
    return np.concatenate([to_input(Image.fromarray(images[i])) for i in indices])


class _Calibration(CalibrationDataReader):  # type: ignore[misc]
    def __init__(self, images: np.ndarray, n: int, seed: int, batch: int = 10) -> None:
        rng = np.random.default_rng(seed)
        picked = rng.choice(len(images), size=min(n, len(images)), replace=False)
        self._batches: Iterator[dict[str, np.ndarray]] = iter(
            {"input": _inputs(images, picked[s : s + batch])}
            for s in range(0, len(picked), batch)
        )

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._batches, None)


def quantize_classifier(
    fp32_path: Path,
    out_path: Path,
    calibration_images: np.ndarray,
    n_calibration: int = 1000,
    seed: int = 0,
    activation_type: QuantType = QuantType.QUInt8,
    reduce_range: bool = True,
) -> Path:
    """Write the int8 model to `out_path` and return it.

    Args:
        fp32_path: the exported fp32 classifier.
        out_path: where to write the quantized model.
        calibration_images: (N, 48, 48) uint8 training faces to calibrate on.
        n_calibration: how many random faces to calibrate with.
        seed: seed of that random sample.
        activation_type: uint8 (default) or int8 activations.
        reduce_range: 7-bit quantization range. Keep it on: without it, CPUs that have AVX2
            but no VNNI (many cloud machines, GitHub runners) saturate an int16 intermediate in
            ONNX Runtime and the model drifts badly (probabilities off by 0.26 on the CI runner);
            with it the result is bit-identical across CPUs.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        quant_pre_process(str(fp32_path), str(prepared))
        # Softmax would otherwise be quantized too, leaving `probs` on a 1/256 grid.
        keep_float = [
            n.name
            for n in onnx.load(str(prepared)).graph.node
            if n.op_type == "Softmax"
        ]
        quantize_static(
            str(prepared),
            str(out_path),
            _Calibration(calibration_images, n_calibration, seed),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=activation_type,
            reduce_range=reduce_range,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=keep_float,
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )
    return out_path


def _logits(path: Path, images: np.ndarray, batch: int = 50) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = [
        sess.run(
            ["logits"],
            {"input": _inputs(images, np.arange(s, min(s + batch, len(images))))},
        )[0]
        for s in range(0, len(images), batch)
    ]
    return np.concatenate(out)


def verify_quantized(
    fp32_path: Path,
    int8_path: Path,
    images: np.ndarray,
    labels: np.ndarray,
    min_agreement: float = 0.95,
    max_accuracy_drop: float = 0.01,
) -> dict[str, float]:
    """Compare int8 with fp32 on held-out faces; raise `QuantizationError` on drift."""
    ref = _logits(fp32_path, images).argmax(axis=1)
    got = _logits(int8_path, images).argmax(axis=1)
    report = {
        "agreement": float((ref == got).mean()),
        "accuracy_fp32": float((ref == labels).mean()),
        "accuracy_int8": float((got == labels).mean()),
    }
    if report["agreement"] < min_agreement:
        raise QuantizationError(
            f"int8 agrees with fp32 on {report['agreement']:.3f} of images, need {min_agreement}"
        )
    if report["accuracy_fp32"] - report["accuracy_int8"] > max_accuracy_drop:
        raise QuantizationError(
            f"int8 loses {report['accuracy_fp32'] - report['accuracy_int8']:.3f} accuracy, limit {max_accuracy_drop}"
        )
    return report
