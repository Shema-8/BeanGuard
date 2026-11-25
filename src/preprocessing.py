"""Data acquisition helpers, preprocessing and dataset analytics.

TensorFlow is imported lazily so the API can start (and the unit tests can run)
without paying the TF import cost.
"""
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src import config


# --------------------------------------------------------------------------
# Single-image preprocessing (used by the API at prediction time)
# --------------------------------------------------------------------------
def is_valid_image(data: bytes) -> bool:
    """True if the bytes decode as an image Pillow understands."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return False


def preprocess_bytes(data: bytes) -> np.ndarray:
    """bytes -> float32 array of shape (1, 224, 224, 3), pixel range 0-255.

    Scaling to [-1, 1] happens *inside* the model (Rescaling layer), so the
    training and serving paths cannot drift apart.
    """
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)          # phone photos are often rotated
        im = im.convert("RGB").resize(config.IMG_SIZE, Image.BILINEAR)
        arr = np.asarray(im, dtype="float32")
    return np.expand_dims(arr, 0)


# --------------------------------------------------------------------------
# Dataset pipeline (used by training and retraining)
# --------------------------------------------------------------------------
def list_classes(directory: Path = None) -> list[str]:
    directory = Path(directory or config.TRAIN_DIR)
    return sorted(p.name for p in directory.iterdir() if p.is_dir())


def make_dataset(directory, class_names, validation_split=None, subset=None, shuffle=True):
    """tf.data pipeline: decode -> resize -> batch -> prefetch."""
    import tensorflow as tf

    ds = tf.keras.utils.image_dataset_from_directory(
        str(directory),
        labels="inferred",
        label_mode="int",
        class_names=list(class_names),
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        seed=config.SEED,
        validation_split=validation_split,
        subset=subset,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def count_images(directory) -> dict[str, int]:
    counts = {}
    for cls in sorted(p for p in Path(directory).iterdir() if p.is_dir()):
        counts[cls.name] = sum(1 for f in cls.rglob("*") if f.suffix.lower() in config.ALLOWED_EXT)
    return counts


def compute_class_weights(directory, class_names) -> dict[int, float]:
    """Balanced class weights: n_samples / (n_classes * n_class_samples)."""
    counts = count_images(directory)
    total = sum(counts.get(c, 0) for c in class_names)
    n_cls = len(class_names)
    return {i: total / (n_cls * max(counts.get(c, 0), 1)) for i, c in enumerate(class_names)}


# --------------------------------------------------------------------------
# Dataset analytics for the dashboard ("what story does the data tell?")
# --------------------------------------------------------------------------
def image_features(path: Path) -> dict | None:
    """Cheap per-image features: size, aspect ratio, brightness, contrast, saturation."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            small = im.convert("RGB").resize((96, 96))
            gray = np.asarray(small.convert("L"), dtype="float32")
            sat = np.asarray(small.convert("HSV"), dtype="float32")[..., 1]
        return {
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 3),
            "brightness": round(float(gray.mean()), 2),
            "contrast": round(float(gray.std()), 2),
            "saturation": round(float(sat.mean()), 2),
        }
    except (OSError, UnidentifiedImageError, ValueError):
        return None


def dataset_summary(directory, max_per_class: int = 80) -> dict:
    """Class counts + per-image feature table (sampled) + auto-generated interpretations."""
    directory = Path(directory)
    counts = count_images(directory)
    rows = []
    for cls in sorted(p for p in directory.iterdir() if p.is_dir()):
        files = sorted(f for f in cls.rglob("*") if f.suffix.lower() in config.ALLOWED_EXT)
        step = max(len(files) // max_per_class, 1)
        for f in files[::step][:max_per_class]:
            feats = image_features(f)
            if feats:
                rows.append({"class": cls.name, **feats})
    return {"class_counts": counts, "features": rows, "interpretations": interpret(counts, rows)}


def interpret(counts: dict, rows: list[dict]) -> list[dict]:
    """Turns the numbers into plain-English findings (at least 3 features)."""
    out = []
    if counts:
        big, small = max(counts, key=counts.get), min(counts, key=counts.get)
        ratio = counts[big] / max(counts[small], 1)
        out.append({
            "feature": "Class balance",
            "finding": (f"'{big}' has {counts[big]} images and '{small}' only {counts[small]} "
                        f"(imbalance ratio {ratio:.1f}x)."),
            "implication": ("Balanced class weights are applied during training so rare classes are "
                            "not ignored; watch per-class recall rather than accuracy alone."
                            if ratio > 1.5 else
                            "The dataset is well balanced, so accuracy is a fair headline metric."),
        })
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        g = df.groupby("class")[["brightness", "contrast", "saturation", "aspect_ratio"]].mean()
        for col, label, why in [
            ("brightness", "Brightness", "Lighting differences between classes can let a model 'cheat' on "
                                         "brightness instead of looking at the leaf; random contrast "
                                         "augmentation reduces that shortcut."),
            ("saturation", "Colour saturation", "Diseased leaves often show colour changes (yellowing, brown or "
                                                "rust-coloured spots), so saturation helps separate classes; "
                                                "healthy leaves are uniformly green."),
            ("aspect_ratio", "Aspect ratio", "Portrait vs landscape photos are squashed to 224x224, so strongly "
                                             "non-square classes may be slightly distorted."),
        ]:
            hi, lo = g[col].idxmax(), g[col].idxmin()
            out.append({
                "feature": label,
                "finding": (f"Highest mean {col.replace('_', ' ')} in '{hi}' ({g[col].max():.1f}); "
                            f"lowest in '{lo}' ({g[col].min():.1f})."),
                "implication": why,
            })
    return out
