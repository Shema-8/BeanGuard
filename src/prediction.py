"""Prediction service. Hot-reloads the model whenever the file on disk changes,
so every API replica picks up a freshly retrained model without a restart."""
import json
import threading
import time

import numpy as np

from src import config
from src.preprocessing import preprocess_bytes


class ModelNotAvailable(RuntimeError):
    pass


def model_info() -> dict:
    """Version + test metrics of the model currently on disk (from metrics.json)."""
    if not config.METRICS_PATH.exists():
        return {}
    return json.loads(config.METRICS_PATH.read_text())


class Predictor:
    def __init__(self):
        self._model = None
        self._mtime = None
        self._lock = threading.Lock()
        self.class_names: list[str] = []
        self.version = 0

    def _ensure_loaded(self):
        if not config.MODEL_PATH.exists() or not config.CLASS_NAMES_PATH.exists():
            raise ModelNotAvailable("No trained model found. Train one (notebook or `python -m src.train`).")
        mtime = config.MODEL_PATH.stat().st_mtime
        if self._model is None or mtime != self._mtime:
            with self._lock:
                if self._model is None or mtime != self._mtime:
                    import tensorflow as tf
                    self._model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
                    self.class_names = json.loads(config.CLASS_NAMES_PATH.read_text())
                    self.version = int(model_info().get("version", 1))
                    self._mtime = mtime

    def predict(self, image_bytes: bytes) -> dict:
        self._ensure_loaded()
        start = time.perf_counter()
        x = preprocess_bytes(image_bytes)
        probs = np.asarray(self._model(x, training=False))[0]
        order = probs.argsort()[::-1]
        top = [{"label": self.class_names[i], "confidence": round(float(probs[i]), 4)} for i in order[:3]]
        latency_ms = (time.perf_counter() - start) * 1000
        best = top[0]
        return {
            "label": best["label"],
            "confidence": best["confidence"],
            "uncertain": best["confidence"] < config.CONFIDENCE_THRESHOLD,
            "top3": top,
            "model_version": self.version,
            "latency_ms": round(latency_ms, 2),
        }


predictor = Predictor()
