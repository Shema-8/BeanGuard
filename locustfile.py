
from locust import HttpUser, task, between
from pathlib import Path
import random


class BeanGuardUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        # Find sample images in the local test dataset.
        test_dir = Path("data/test")
        self.images = list(test_dir.glob("*/*.jpg"))
        self.images += list(test_dir.glob("*/*.jpeg"))
        self.images += list(test_dir.glob("*/*.png"))

    @task(4)
    def predict_image(self):
        if not self.images:
            self.environment.runner.quit()
            return

        image_path = random.choice(self.images)

        with open(image_path, "rb") as image:
            self.client.post(
                "/predict",
                files={
                    "file": (
                        image_path.name,
                        image,
                        "image/jpeg"
                    )
                },
                name="/predict"
            )

    @task(1)
    def check_health(self):
        self.client.get("/health")
