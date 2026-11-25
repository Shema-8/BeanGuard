"""Central configuration.

Every path and threshold can be overridden with an environment variable, which is
what the Docker / cloud deployment does.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---- Data ---------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "heritage.db"))

# ---- Model artefacts ----------------------------------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
ARCHIVE_DIR = MODEL_DIR / "archive"
MODEL_PATH = MODEL_DIR / "beanguard_mobilenetv2.keras"
MODEL_H5_PATH = MODEL_DIR / "beanguard_mobilenetv2.h5"      # best-effort .h5 export of the same model
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
RETRAIN_STATUS_PATH = MODEL_DIR / "retrain_status.json"
RETRAIN_LOCK_PATH = MODEL_DIR / ".retrain.lock"

# ---- Image / training ---------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
SEED = 42
USE_IMAGENET_WEIGHTS = os.getenv("USE_IMAGENET_WEIGHTS", "1") == "1"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---- Serving ------------------------------------------------------------
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.50))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 50))

# ---- Retraining ---------------------------------------------------------
# Automatic trigger: retrain once this many *new labelled* images are waiting.
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", 30))
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", 8))
RETRAIN_LR = float(os.getenv("RETRAIN_LR", 1e-5))
# A candidate model is promoted only if macro-F1 on the untouched test set is
# at least (old macro-F1 - tolerance).
PROMOTION_TOLERANCE = float(os.getenv("PROMOTION_TOLERANCE", 0.0))
