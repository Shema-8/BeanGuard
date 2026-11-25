import json

from fastapi.testclient import TestClient

from api.main import app
from tests.conftest import make_jpeg


def test_health_and_503_without_model(tmp_env):
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/predict", files={"file": ("x.jpg", make_jpeg(), "image/jpeg")})
        assert r.status_code == 503
        assert "X-Served-By" in r.headers


def test_predict_rejects_non_image(tmp_env):
    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("x.jpg", b"nope", "image/jpeg")})
        assert r.status_code == 400


def test_upload_validates_label_and_stores(tmp_env):
    (tmp_env / "models").mkdir()
    (tmp_env / "models" / "class_names.json").write_text(json.dumps(["healthy", "bean_rust"]))
    with TestClient(app) as client:
        bad = client.post("/upload", data={"label": "pizza"},
                          files=[("files", ("a.jpg", make_jpeg(), "image/jpeg"))])
        assert bad.status_code == 400
        ok = client.post("/upload", data={"label": "Healthy"},
                         files=[("files", ("a.jpg", make_jpeg(), "image/jpeg")),
                                ("files", ("b.jpg", make_jpeg((1, 2, 3)), "image/jpeg")),
                                ("files", ("bad.jpg", b"junk", "image/jpeg"))])
        body = ok.json()
        assert body["accepted"] == 2 and body["rejected_count"] == 1
        assert body["pending_total"] == 2
        dup = client.post("/upload", data={"label": "healthy"},
                          files=[("files", ("a.jpg", make_jpeg(), "image/jpeg"))]).json()
        assert dup["accepted"] == 0
