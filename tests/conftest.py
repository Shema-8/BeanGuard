import io

import numpy as np
import pytest
from PIL import Image

from src import config


@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    """Point every path at a temp dir so tests never touch real data/models."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "TRAIN_DIR", tmp_path / "data" / "train")
    monkeypatch.setattr(config, "TEST_DIR", tmp_path / "data" / "test")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "data" / "uploads")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "m.h5")
    monkeypatch.setattr(config, "CLASS_NAMES_PATH", tmp_path / "models" / "class_names.json")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "models" / "metrics.json")
    monkeypatch.setattr(config, "RETRAIN_STATUS_PATH", tmp_path / "models" / "retrain_status.json")
    monkeypatch.setattr(config, "RETRAIN_LOCK_PATH", tmp_path / "models" / ".lock")
    return tmp_path


def make_jpeg(color=(200, 50, 50), size=(64, 48)) -> bytes:
    arr = np.full((size[1], size[0], 3), color, dtype="uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()
