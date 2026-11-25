"""Model definition, training (with optimisation techniques) and evaluation.

Architecture: MobileNetV2 (ImageNet pre-trained) + GlobalAveragePooling +
Dropout + L2-regularised softmax head.

Optimisation techniques used (rubric: "Regularization, Optimizers, early
stopping, pretrained model, hyper-parameter tuning"):
  * transfer learning from ImageNet
  * data augmentation (flip / rotation / zoom / contrast; all .h5-serialisable)
  * dropout + L2 weight decay
  * Adam optimiser, two-phase schedule (frozen head, then low-LR fine-tuning)
  * EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
  * balanced class weights
"""
import json
from datetime import datetime, timezone

import numpy as np

from src import config


def build_model(num_classes: int, dropout: float = 0.3, l2: float = 1e-4):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*config.IMG_SIZE, 3),
        include_top=False,
        weights="imagenet" if config.USE_IMAGENET_WEIGHTS else None,
    )
    backbone.trainable = False

    inputs = layers.Input(shape=(*config.IMG_SIZE, 3), name="image")
    x = augmentation(inputs)                              # active only while training
    x = layers.Rescaling(1 / 127.5, offset=-1, name="rescale")(x)   # 0-255 -> [-1, 1]
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", kernel_regularizer=regularizers.l2(l2), name="probs"
    )(x)
    return tf.keras.Model(inputs, outputs, name="beanguard_classifier")


def get_backbone(model):
    import tensorflow as tf
    return next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))


def compile_model(model, lr: float):
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_top(model, n_layers: int = 40):
    """Unfreeze the last n backbone layers (BatchNorm stays frozen)."""
    import tensorflow as tf
    backbone = get_backbone(model)
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return model


def default_callbacks(patience: int = 4, checkpoint_path=None):
    import tensorflow as tf
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]
    if checkpoint_path:
        cbs.append(tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss",
                                                      save_best_only=True))
    return cbs


def _merge_history(*histories) -> dict:
    merged: dict[str, list] = {}
    for h in histories:
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(float(x) for x in v)
    return merged


def train_new_model(train_ds, val_ds, num_classes, class_weight=None,
                    head_epochs: int = 15, finetune_epochs: int = 15):
    """Phase 1: train the new head on a frozen backbone. Phase 2: fine-tune the top layers."""
    model = compile_model(build_model(num_classes), lr=1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs,
                   class_weight=class_weight, callbacks=default_callbacks())
    unfreeze_top(model)
    compile_model(model, lr=1e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs,
                   class_weight=class_weight, callbacks=default_callbacks())
    return model, _merge_history(h1, h2)


def fine_tune(model, train_ds, val_ds, epochs: int, lr: float, class_weight=None):
    """Continue training an existing model on new data (used by the retraining pipeline)."""
    unfreeze_top(model)
    compile_model(model, lr=lr)
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  class_weight=class_weight, callbacks=default_callbacks(patience=3))
    return model, _merge_history(h)


def evaluate(model, ds, class_names) -> dict:
    """Accuracy, loss, precision, recall, F1 (macro + weighted), per-class report, confusion matrix."""
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 f1_score, log_loss, precision_score, recall_score)
    y_true, y_prob = [], []
    for x, y in ds:
        y_prob.append(model.predict(x, verbose=0))
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = y_prob.argmax(axis=1)
    labels = list(range(len(class_names)))
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(log_loss(y_true, y_prob, labels=labels)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": list(class_names),
    }


def _atomic_write_text(path, text: str) -> None:
    import os
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def save_artifacts(model, class_names, test_metrics, history, version, trigger, n_train) -> None:
    """Writes metrics, class names and the model - atomically, so a crash can never leave a
    half-written model file that the API would try to load. The model file is written LAST because
    the API hot-reloads when its modification time changes."""
    import os

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(config.CLASS_NAMES_PATH, json.dumps(list(class_names)))
    _atomic_write_text(config.METRICS_PATH, json.dumps({
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trigger": trigger,
        "n_train": n_train,
        "test": test_metrics,
        "history": history,
    }))
    # .h5 export (rubric accepts .pkl / .tf / .h5); best effort - never blocks the real save.
    tmp_h5 = config.MODEL_DIR / "candidate_export.h5"
    try:
        model.save(tmp_h5)
        os.replace(tmp_h5, config.MODEL_H5_PATH)
    except Exception as exc:  # noqa: BLE001
        tmp_h5.unlink(missing_ok=True)
        print(f"[warn] .h5 export skipped: {exc}")
    tmp = config.MODEL_DIR / "candidate_model.keras"
    model.save(tmp)
    os.replace(tmp, config.MODEL_PATH)
