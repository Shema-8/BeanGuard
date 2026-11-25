"""Load test for the BeanGuard classifier.

Interactive:  locust -f locust/locustfile.py --host http://localhost
Headless:     see locust/run_experiments.sh
"""
import glob
import io
import os
import random

from locust import HttpUser, between, task
from PIL import Image
import numpy as np

IMAGE_GLOB = os.getenv("LOCUST_IMAGES", "data/test/**/*.jpg")


def _load_images(limit: int = 30) -> list[tuple[str, bytes]]:
    paths = glob.glob(IMAGE_GLOB, recursive=True)
    random.shuffle(paths)
    images = []
    for p in paths[:limit]:
        with open(p, "rb") as f:
            images.append((os.path.basename(p), f.read()))
    if not images:                                  # fall back to synthetic images
        for i in range(5):
            arr = np.random.randint(0, 255, (224, 224, 3), dtype="uint8")
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG")
            images.append((f"synthetic_{i}.jpg", buf.getvalue()))
    return images


IMAGES = _load_images()          # read once so disk I/O does not skew latency


class BeanUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGES)
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")},
                              name="/predict", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif "label" not in resp.json():
                resp.failure("no label in response")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")

    @task(1)
    def status(self):
        self.client.get("/status", name="/status")
