"""SQLite persistence for uploads (retraining data), model versions and prediction logs."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    label TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    uploaded_at TEXT NOT NULL,
    used_in_version INTEGER
);
CREATE TABLE IF NOT EXISTS model_versions (
    version INTEGER PRIMARY KEY,
    trained_at TEXT NOT NULL,
    trigger TEXT,
    n_train INTEGER,
    n_new INTEGER,
    accuracy REAL,
    f1_macro REAL,
    prev_f1_macro REAL,
    promoted INTEGER,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    label TEXT,
    confidence REAL,
    latency_ms REAL,
    model_version INTEGER,
    uncertain INTEGER,
    host TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


# ---- uploads --------------------------------------------------------------
def add_upload(filename: str, label: str, path: str) -> bool:
    """Returns False if this exact file path was already stored."""
    with connect() as c:
        cur = c.execute(
            "INSERT OR IGNORE INTO uploads (filename, label, path, uploaded_at) VALUES (?,?,?,?)",
            (filename, label, path, now()),
        )
        return cur.rowcount > 0


def pending_uploads() -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM uploads WHERE used_in_version IS NULL ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def pending_count() -> int:
    with connect() as c:
        return c.execute("SELECT COUNT(*) FROM uploads WHERE used_in_version IS NULL").fetchone()[0]


def pending_by_label() -> dict:
    with connect() as c:
        rows = c.execute(
            "SELECT label, COUNT(*) n FROM uploads WHERE used_in_version IS NULL GROUP BY label"
        ).fetchall()
    return {r["label"]: r["n"] for r in rows}


def mark_uploads_used(ids: list[int], version: int) -> None:
    with connect() as c:
        c.executemany("UPDATE uploads SET used_in_version=? WHERE id=?", [(version, i) for i in ids])


# ---- model versions -------------------------------------------------------
def latest_version() -> int:
    with connect() as c:
        v = c.execute("SELECT MAX(version) FROM model_versions").fetchone()[0]
    return v or 0


def add_model_version(version, trigger, n_train, n_new, accuracy, f1_macro,
                      prev_f1_macro, promoted, notes="") -> None:
    with connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?,?,?,?,?,?)",
            (version, now(), trigger, n_train, n_new, accuracy, f1_macro,
             prev_f1_macro, int(promoted), notes),
        )


def list_versions() -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM model_versions ORDER BY version").fetchall()
    return [dict(r) for r in rows]


# ---- predictions / monitoring --------------------------------------------
def log_prediction(label, confidence, latency_ms, model_version, uncertain, host) -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO predictions (ts,label,confidence,latency_ms,model_version,uncertain,host)"
            " VALUES (?,?,?,?,?,?,?)",
            (now(), label, confidence, latency_ms, model_version, int(uncertain), host),
        )


def recent_predictions(limit: int = 200) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[int(round(q * (len(sorted_vals) - 1)))]


def prediction_stats() -> dict:
    with connect() as c:
        total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        rows = c.execute(
            "SELECT latency_ms, uncertain, label FROM predictions ORDER BY id DESC LIMIT 2000"
        ).fetchall()
    lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
    per_class: dict[str, int] = {}
    for r in rows:
        per_class[r["label"]] = per_class.get(r["label"], 0) + 1
    return {
        "total_predictions": total,
        "avg_latency_ms": round(sum(lat) / len(lat), 2) if lat else 0.0,
        "p50_latency_ms": round(_pct(lat, 0.50), 2),
        "p95_latency_ms": round(_pct(lat, 0.95), 2),
        "uncertain_share": round(sum(r["uncertain"] for r in rows) / len(rows), 4) if rows else 0.0,
        "predictions_per_class": per_class,
    }
