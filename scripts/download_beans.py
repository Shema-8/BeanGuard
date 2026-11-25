
"""Download the Makerere AI Lab bean leaf dataset."""

import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

URLS = {
    "train": "https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/train.zip",
    "validation": "https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/validation.zip",
    "test": "https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/test.zip",
}

OUT = Path("raw_images")


def main():
    tmp = Path(tempfile.mkdtemp())

    try:
        for name, url in URLS.items():
            zpath = tmp / f"{name}.zip"

            print(f"Downloading {name}...")

            with requests.get(
                url,
                stream=True,
                timeout=180,
            ) as response:
                response.raise_for_status()

                with open(zpath, "wb") as file:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            file.write(chunk)

            print(f"Extracting {name}...")

            with zipfile.ZipFile(zpath) as archive:
                archive.extractall(tmp / name)

        OUT.mkdir(parents=True, exist_ok=True)

        count = 0

        for img in tmp.rglob("*"):
            if img.is_file() and img.suffix.lower() in [".jpg", ".jpeg"]:
                split = img.relative_to(tmp).parts[0]
                class_name = img.parent.name

                dest = OUT / class_name
                dest.mkdir(parents=True, exist_ok=True)

                shutil.copy2(
                    img,
                    dest / f"{split}_{img.name}",
                )

                count += 1

        print(f"\nSuccessfully downloaded {count} images.")

        for folder in sorted(OUT.iterdir()):
            if folder.is_dir():
                total = sum(
                    1 for path in folder.iterdir()
                    if path.is_file()
                )
                print(f"{folder.name:<25} {total}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
