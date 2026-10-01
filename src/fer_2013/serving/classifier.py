"""ONNX classifier session: probabilities plus layer4 feature maps."""

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
        self.fc_weight: np.ndarray = np.load(fc_weight_path)
        self.sha: str = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:8]

    def predict(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,3,224,224) float32 -> softmax probs (N,7) and features (N,512,7,7)."""
        logits, features = self._sess.run(None, {"input": batch})
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32), features
