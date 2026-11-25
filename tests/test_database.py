from src import database as db


def test_upload_lifecycle(tmp_env):
    db.init_db()
    assert db.add_upload("a.jpg", "healthy", "/x/a.jpg")
    assert not db.add_upload("a.jpg", "healthy", "/x/a.jpg")      # same path ignored
    db.add_upload("b.jpg", "bean_rust", "/x/b.jpg")
    assert db.pending_count() == 2
    assert db.pending_by_label() == {"healthy": 1, "bean_rust": 1}
    db.mark_uploads_used([r["id"] for r in db.pending_uploads()], 2)
    assert db.pending_count() == 0


def test_prediction_stats(tmp_env):
    db.init_db()
    for ms in (10, 20, 30, 40, 100):
        db.log_prediction("healthy", 0.9, ms, 1, False, "host")
    s = db.prediction_stats()
    assert s["total_predictions"] == 5
    assert s["avg_latency_ms"] == 40.0
    assert s["p95_latency_ms"] == 100
