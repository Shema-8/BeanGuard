import numpy as np

from src import config
from src.preprocessing import (compute_class_weights, dataset_summary, interpret, is_valid_image,
                               preprocess_bytes)
from tests.conftest import make_jpeg


def test_valid_and_invalid_image():
    assert is_valid_image(make_jpeg())
    assert not is_valid_image(b"definitely not an image")


def test_preprocess_shape_and_range():
    arr = preprocess_bytes(make_jpeg(size=(300, 120)))
    assert arr.shape == (1, *config.IMG_SIZE, 3)
    assert arr.dtype == np.float32 and 0 <= arr.min() and arr.max() <= 255


def _fill(root, counts):
    for cls, n in counts.items():
        (root / cls).mkdir(parents=True)
        for i in range(n):
            (root / cls / f"{i}.jpg").write_bytes(make_jpeg())


def test_class_weights_favour_rare_class(tmp_path):
    _fill(tmp_path, {"a": 30, "b": 10})
    w = compute_class_weights(tmp_path, ["a", "b"])
    assert w[1] > w[0]


def test_dataset_summary_and_interpretation(tmp_path):
    _fill(tmp_path, {"a": 6, "b": 3})
    s = dataset_summary(tmp_path)
    assert s["class_counts"] == {"a": 6, "b": 3}
    assert len(s["features"]) == 9
    features = {i["feature"] for i in s["interpretations"]}
    assert {"Class balance", "Brightness", "Colour saturation", "Aspect ratio"} <= features
    assert interpret({}, []) == []
