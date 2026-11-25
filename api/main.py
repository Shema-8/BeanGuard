"""FastAPI service: prediction, bulk upload, retraining trigger, monitoring and data insights."""
import hashlib
import io
import json
import socket
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import PurePosixPath

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src import config, database as db
from src.prediction import ModelNotAvailable, model_info, predictor
from src.preprocessing import dataset_summary, is_valid_image
from src.retrain import get_status, maybe_auto_trigger, start_retrain_async

HOST = socket.gethostname()
START_TIME = time.time()
_insight_cache = {"t": 0.0, "data": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()
    # Register the initial model (trained in the notebook) as version 1 if the DB is new.
    info = model_info()
    if info and db.latest_version() == 0:
        t = info.get("test", {})
        db.add_model_version(info.get("version", 1), info.get("trigger", "initial"),
                             info.get("n_train"), 0, t.get("accuracy"), t.get("f1_macro"),
                             None, True, "initial model")
    yield


app = FastAPI(
    title="BeanGuard API",
    description="Bean leaf disease classification with a full MLOps loop: "
                "predict, upload, retrain, monitor.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def add_host_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Served-By"] = HOST       # shows load-balancing across replicas
    return response


def _too_big(data: bytes) -> bool:
    return len(data) > config.MAX_UPLOAD_MB * 1024 * 1024


# --------------------------------------------------------------------------- health
@app.get("/health", tags=["monitoring"])
def health():
    return {"status": "ok", "host": HOST, "uptime_seconds": round(time.time() - START_TIME, 1),
            "model_loaded": config.MODEL_PATH.exists()}


@app.get("/status", tags=["monitoring"])
def status():
    info = model_info()
    return {
        "host": HOST,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_available": config.MODEL_PATH.exists(),
        "model_version": info.get("version"),
        "model_trained_at": info.get("trained_at"),
        "classes": json.loads(config.CLASS_NAMES_PATH.read_text()) if config.CLASS_NAMES_PATH.exists() else [],
        "pending_uploads": db.pending_count(),
        "pending_by_label": db.pending_by_label(),
        "retrain_threshold": config.RETRAIN_THRESHOLD,
        "retrain": get_status(),
    }


@app.get("/metrics", tags=["monitoring"])
def metrics():
    """Model quality (test set) + live serving statistics."""
    info = model_info()
    return {"model": {"version": info.get("version"), "trained_at": info.get("trained_at"),
                      "n_train": info.get("n_train"), "test": info.get("test"),
                      "history": info.get("history")},
            "serving": db.prediction_stats(),
            "versions": db.list_versions()}


@app.get("/predictions/recent", tags=["monitoring"])
def recent(limit: int = 200):
    return db.recent_predictions(min(limit, 1000))


# --------------------------------------------------------------------------- predict
@app.post("/predict", tags=["prediction"])
async def predict(file: UploadFile = File(...)):
    """Classify ONE image."""
    data = await file.read()
    if not data or _too_big(data):
        raise HTTPException(413, "Empty or too large file.")
    if not is_valid_image(data):
        raise HTTPException(400, "File is not a valid image.")
    try:
        result = await run_in_threadpool(predictor.predict, data)
    except ModelNotAvailable as exc:
        raise HTTPException(503, str(exc))
    db.log_prediction(result["label"], result["confidence"], result["latency_ms"],
                      result["model_version"], result["uncertain"], HOST)
    result["served_by"] = HOST
    return result


# --------------------------------------------------------------------------- upload
def _canonical_label(raw: str, classes: list[str]) -> str | None:
    lookup = {c.lower(): c for c in classes}
    return lookup.get((raw or "").strip().lower())


def _store(data: bytes, filename: str, label: str) -> tuple[bool, str]:
    if not is_valid_image(data):
        return False, "not a valid image"
    digest = hashlib.sha1(data).hexdigest()[:12]
    ext = PurePosixPath(filename).suffix.lower()
    if ext not in config.ALLOWED_EXT:
        return False, f"unsupported extension '{ext}'"
    dest = config.UPLOAD_DIR / label / f"{digest}{ext}"
    if dest.exists():
        return False, "duplicate"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    db.add_upload(PurePosixPath(filename).name, label, str(dest))
    return True, "ok"


@app.post("/upload", tags=["retraining"])
async def upload(files: list[UploadFile] = File(...), label: str | None = Form(None)):
    """Bulk upload labelled images for retraining.

    * Images: send them with a `label` form field (must be an existing class).
    * ZIP: one folder per class (e.g. healthy/img1.jpg); no `label` needed.
    """
    if not config.CLASS_NAMES_PATH.exists():
        raise HTTPException(503, "No trained model/classes yet.")
    classes = json.loads(config.CLASS_NAMES_PATH.read_text())
    fixed = None
    if label:
        fixed = _canonical_label(label, classes)
        if fixed is None:
            raise HTTPException(400, f"Unknown label '{label}'. Valid labels: {classes}")

    accepted, rejected = 0, []
    for f in files:
        data = await f.read()
        if _too_big(data):
            rejected.append({"file": f.filename, "reason": "file too large"})
            continue
        if (f.filename or "").lower().endswith(".zip"):
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                rejected.append({"file": f.filename, "reason": "bad zip"})
                continue
            members = [m for m in zf.infolist() if not m.is_dir()][:2000]
            for m in members:
                parts = [p for p in PurePosixPath(m.filename).parts if p not in ("/", "..")]
                if "__MACOSX" in parts or len(parts) < 2 or m.file_size > config.MAX_UPLOAD_MB * 1024 * 1024:
                    continue
                lab = fixed or _canonical_label(parts[-2], classes)
                if lab is None:
                    rejected.append({"file": m.filename, "reason": f"folder '{parts[-2]}' is not a known class"})
                    continue
                ok, why = _store(zf.read(m), parts[-1], lab)
                accepted += ok
                if not ok:
                    rejected.append({"file": m.filename, "reason": why})
        else:
            if fixed is None:
                rejected.append({"file": f.filename, "reason": "label is required for image uploads"})
                continue
            ok, why = _store(data, f.filename or "image.jpg", fixed)
            accepted += ok
            if not ok:
                rejected.append({"file": f.filename, "reason": why})

    auto = maybe_auto_trigger() if accepted else False
    return {"accepted": accepted, "rejected": rejected[:50], "rejected_count": len(rejected),
            "pending_total": db.pending_count(), "auto_retrain_started": auto}


# --------------------------------------------------------------------------- retrain
@app.post("/retrain", tags=["retraining"])
def retrain():
    """Manual trigger (the dashboard button)."""
    if not config.MODEL_PATH.exists():
        raise HTTPException(503, "No model to retrain.")
    started = start_retrain_async("manual")
    if not started:
        st = get_status()
        if st.get("state") == "running":
            raise HTTPException(409, "A retraining job is already running.")
        raise HTTPException(400, st.get("message", "Nothing to retrain."))
    return {"started": True, "status": get_status()}


@app.get("/retrain/status", tags=["retraining"])
def retrain_status():
    return get_status()


# --------------------------------------------------------------------------- insights
@app.get("/insights", tags=["visualisation"])
def insights():
    """Data behind the dashboard visualisations (cached for 60 s)."""
    now = time.time()
    if _insight_cache["data"] is None or now - _insight_cache["t"] > 60:
        if not config.TRAIN_DIR.exists() or not any(config.TRAIN_DIR.iterdir()):
            raise HTTPException(404, "No training data found in data/train.")
        _insight_cache["data"] = dataset_summary(config.TRAIN_DIR)
        _insight_cache["t"] = now
    return _insight_cache["data"]
