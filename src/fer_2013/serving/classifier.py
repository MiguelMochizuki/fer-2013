"""ONNX classifier session: calibrated probabilities plus layer4 feature maps."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort


class Classifier:
    """Runs the exported ResNet18. Safe to call from several threads."""

    def __init__(self, model_path: Path, fc_weight_path: Path, threads: int = 2):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self._outputs = [o.name for o in self._sess.get_outputs()]
        meta = self._sess.get_modelmeta().custom_metadata_map
        # Calibration temperature stored by the exporter; None for models that
        # predate calibration (they expose only logits and features).
        self.temperature: float | None = (
            float(meta["temperature"]) if "temperature" in meta else None
        )
        self.fc_weight: np.ndarray = np.load(fc_weight_path)
        self.sha: str = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:8]

    def predict(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,3,224,224) float32 -> probs (N,7) and features (N,512,7,7).

        Probabilities come from the model's calibrated `probs` output when it
        has one, otherwise from a plain softmax of the logits.
        """
        out = dict(
            zip(self._outputs, self._sess.run(None, {"input": batch}), strict=True)
        )
        if "probs" in out:
            probs = out["probs"]
        else:
            logits = out["logits"]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
        return probs.astype(np.float32), out["features"]
