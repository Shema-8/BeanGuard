"""Retraining pipeline.

  1. collect labelled uploads that have not been used yet (stored in the DB)
  2. merge them into data/train/<label>/
  3. preprocess (same tf.data pipeline as the original training)
  4. fine-tune the *existing* model (pre-trained custom model, not ImageNet)
  5. evaluate old vs new on the untouched test set
  6. promote the new model only if it is not worse; archive the old one
  7. log the new version, mark uploads as used; API replicas hot-reload

Triggers: the dashboard button (POST /retrain) or automatically once
RETRAIN_THRESHOLD new labelled images are waiting.
"""
import json
import os
import shutil
import threading
import time
import traceback
from pathlib import Path

from src import config, database as db
from src.model import evaluate, fine_tune, save_artifacts
from src.preprocessing import compute_class_weights, count_images, make_dataset

_STALE_LOCK_SECONDS = 3 * 3600


# ---- status / lock (files on the shared volume so every replica sees them) ----
def set_status(state: str, message: str = "", **extra) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.RETRAIN_STATUS_PATH.write_text(json.dumps(
        {"state": state, "message": message, "updated_at": db.now(), **extra}))


def get_status() -> dict:
    if config.RETRAIN_STATUS_PATH.exists():
        return json.loads(config.RETRAIN_STATUS_PATH.read_text())
    return {"state": "idle", "message": "No retraining has run yet."}


def _acquire_lock() -> bool:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if config.RETRAIN_LOCK_PATH.exists():
        age = time.time() - config.RETRAIN_LOCK_PATH.stat().st_mtime
        if age < _STALE_LOCK_SECONDS:
            return False
        config.RETRAIN_LOCK_PATH.unlink(missing_ok=True)
    try:
        fd = os.open(config.RETRAIN_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _release_lock() -> None:
    config.RETRAIN_LOCK_PATH.unlink(missing_ok=True)


def is_running() -> bool:
    return config.RETRAIN_LOCK_PATH.exists() and get_status().get("state") == "running"


# ---- public entry points ----------------------------------------------------
def start_retrain_async(trigger: str = "manual") -> bool:
    """Starts retraining in a background thread. False if one is already running or nothing to learn from."""
    if db.pending_count() == 0:
        set_status("idle", "Nothing to retrain: no new labelled images uploaded.")
        return False
    if not _acquire_lock():
        return False
    set_status("running", "Starting retraining...", trigger=trigger)
    threading.Thread(target=_run, args=(trigger,), daemon=True).start()
    return True


def maybe_auto_trigger() -> bool:
    if db.pending_count() >= config.RETRAIN_THRESHOLD:
        return start_retrain_async("auto-threshold")
    return False


# ---- the pipeline -------------------------------------------------------------
def _run(trigger: str) -> None:
    try:
        import tensorflow as tf

        class_names = json.loads(config.CLASS_NAMES_PATH.read_text())

        set_status("running", "1/6 Merging uploaded images into the training set", trigger=trigger)
        pending = db.pending_uploads()
        for row in pending:
            dest_dir = config.TRAIN_DIR / row["label"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["path"], dest_dir / Path(row["path"]).name)

        set_status("running", "2/6 Preprocessing and building datasets", trigger=trigger)
        train_ds = make_dataset(config.TRAIN_DIR, class_names, 0.2, "training")
        val_ds = make_dataset(config.TRAIN_DIR, class_names, 0.2, "validation")
        test_ds = make_dataset(config.TEST_DIR, class_names, shuffle=False)
        weights = compute_class_weights(config.TRAIN_DIR, class_names)

        set_status("running", "3/6 Evaluating the current model on the test set", trigger=trigger)
        old_model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
        old_metrics = evaluate(old_model, test_ds, class_names)
        del old_model

        set_status("running", "4/6 Fine-tuning the existing model on the enlarged dataset", trigger=trigger)
        candidate = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
        candidate, history = fine_tune(candidate, train_ds, val_ds, config.RETRAIN_EPOCHS,
                                       config.RETRAIN_LR, weights)

        set_status("running", "5/6 Evaluating the candidate model", trigger=trigger)
        new_metrics = evaluate(candidate, test_ds, class_names)
        promoted = new_metrics["f1_macro"] >= old_metrics["f1_macro"] - config.PROMOTION_TOLERANCE

        set_status("running", "6/6 Saving results", trigger=trigger)
        prev_version = db.latest_version() or 1
        version = prev_version + 1
        n_train = sum(count_images(config.TRAIN_DIR).values())
        if promoted:
            config.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config.MODEL_PATH, config.ARCHIVE_DIR / f"model_v{prev_version}{config.MODEL_PATH.suffix}")
            save_artifacts(candidate, class_names, new_metrics, history, version, trigger, n_train)
        db.add_model_version(version, trigger, n_train, len(pending), new_metrics["accuracy"],
                             new_metrics["f1_macro"], old_metrics["f1_macro"], promoted,
                             "promoted" if promoted else "rejected: worse macro-F1 than current model")
        db.mark_uploads_used([r["id"] for r in pending], version)

        verdict = "PROMOTED" if promoted else "REJECTED (kept the previous model)"
        set_status(
            "completed",
            f"Candidate v{version} {verdict}. Macro-F1 {old_metrics['f1_macro']:.3f} -> "
            f"{new_metrics['f1_macro']:.3f}; accuracy {old_metrics['accuracy']:.3f} -> "
            f"{new_metrics['accuracy']:.3f}.",
            trigger=trigger, promoted=promoted, version=version, new_images=len(pending),
            old_f1=old_metrics["f1_macro"], new_f1=new_metrics["f1_macro"],
            old_accuracy=old_metrics["accuracy"], new_accuracy=new_metrics["accuracy"],
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the dashboard
        traceback.print_exc()
        set_status("failed", f"Retraining failed: {exc}")
    finally:
        _release_lock()
