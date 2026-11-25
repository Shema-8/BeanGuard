"""Turns a raw folder-per-class image collection into the repo's data layout.

    raw_images/
        healthy/*.jpg
        bean_rust/*.jpg
        ...
  ->
    data/train/<class>/   (default 75 %)
    data/test/<class>/    (default 15 %)  never touched by retraining
    data/new_data_demo/<class>/ (default 10 %)  held back to DEMO the retraining upload

Also: validates every file, converts to RGB JPEG (max side 512 px) and removes duplicates.

Usage:  python scripts/split_dataset.py raw_images --out data
"""
import argparse
import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def clean(src: Path, dst: Path) -> bool:
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((512, 512))
            im.save(dst, "JPEG", quality=90)
        return True
    except (OSError, UnidentifiedImageError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("--out", default="data")
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--demo", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw, out = Path(args.raw_dir), Path(args.out)
    rng = random.Random(args.seed)
    seen: set[str] = set()
    summary = {}

    for cls_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        files = sorted(f for f in cls_dir.rglob("*") if f.suffix.lower() in EXT)
        rng.shuffle(files)
        good = []
        tmp = out / "_tmp" / cls_dir.name
        tmp.mkdir(parents=True, exist_ok=True)
        for f in files:
            digest = hashlib.md5(f.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            dst = tmp / f"{cls_dir.name}_{digest[:10]}.jpg"
            if clean(f, dst):
                good.append(dst)
        n = len(good)
        n_test, n_demo = int(n * args.test), int(n * args.demo)
        parts = {"test": good[:n_test], "new_data_demo": good[n_test:n_test + n_demo],
                 "train": good[n_test + n_demo:]}
        for split, items in parts.items():
            d = out / split / cls_dir.name
            d.mkdir(parents=True, exist_ok=True)
            for item in items:
                shutil.move(str(item), d / item.name)
        summary[cls_dir.name] = {k: len(v) for k, v in parts.items()}
    shutil.rmtree(out / "_tmp", ignore_errors=True)

    print(f"{'class':<14}{'train':>7}{'test':>7}{'demo':>7}")
    for c, s in summary.items():
        print(f"{c:<14}{s['train']:>7}{s['test']:>7}{s['new_data_demo']:>7}")


if __name__ == "__main__":
    main()
