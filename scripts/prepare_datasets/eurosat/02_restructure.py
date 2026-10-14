"""Step 02 — Restructure EuroSAT RGB into ImageFolder layout with train/val/test splits.

Reads the raw extracted directory produced by 01_download.py. The source zip
contains one directory per land-use class with no splits:

    EuroSAT_RGB/
        AnnualCrop/       *.jpg   (~3000 images, 64x64)
        Forest/           *.jpg
        HerbaceousVegetation/
        Highway/
        Industrial/
        Pasture/
        PermanentCrop/
        Residential/
        River/
        SeaLake/

This script applies a stratified 70/15/15 split across all 10 classes and copies
images into data/eurosat/<split>/<label>/, then writes a metadata.csv sidecar.

Output layout:
    data/eurosat/
        train/
            AnnualCrop/  *.jpg
            Forest/      *.jpg
            ...
        val/  ...
        test/ ...
        metadata.csv
        README.md  (written by C2_package.py)

Usage:
    python scripts/prepare_datasets/eurosat/02_restructure.py
    python scripts/prepare_datasets/eurosat/02_restructure.py --src data/raw/eurosat/extracted/EuroSAT_RGB --dest data/eurosat
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC = REPO_ROOT / "data" / "raw" / "eurosat" / "extracted" / "EuroSAT_RGB"
DEFAULT_DEST = REPO_ROOT / "data" / "eurosat"


def find_eurosat_root(extracted: Path) -> Path:
    """Handle variations: EuroSAT_RGB/ may be directly under extracted/ or nested."""
    candidate = extracted / "EuroSAT_RGB"
    if candidate.exists():
        return candidate
    # If the zip extracted directly without the wrapper dir, use extracted/ itself
    subdirs = [p for p in extracted.iterdir() if p.is_dir()]
    if subdirs and all(p.name[0].isupper() for p in subdirs):
        return extracted
    raise RuntimeError(
        f"Cannot locate EuroSAT_RGB class directories under {extracted}.\n"
        "Run 01_download.py and inspect the output first."
    )


def iter_valid_images(class_dir: Path) -> list[Path]:
    paths = []
    for p in sorted(class_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            paths.append(p)
        except (UnidentifiedImageError, Exception):
            print(f"  Skipping corrupt image: {p.name}")
    return paths


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )


def restructure(src: Path, dest: Path):
    if not src.exists():
        # Try to locate automatically from the default raw path
        raw_root = REPO_ROOT / "data" / "raw" / "eurosat" / "extracted"
        if raw_root.exists():
            src = find_eurosat_root(raw_root)
        else:
            raise FileNotFoundError(
                f"Source dir not found: {src}\nRun 01_download.py first."
            )

    if dest.exists():
        print(f"Destination already exists: {dest}")
        print("Remove it to re-run. Exiting.")
        return

    class_dirs = sorted([p for p in src.iterdir() if p.is_dir()], key=lambda p: p.name)
    print(f"Found {len(class_dirs)} classes: {[p.name for p in class_dirs]}")

    rows = []
    image_id = 0

    for class_dir in class_dirs:
        label = class_dir.name
        print(f"\nProcessing class: {label}")
        paths = iter_valid_images(class_dir)
        print(f"  Valid images: {len(paths)}")

        train_idx, val_idx, test_idx = split_indices(
            len(paths), TRAIN_RATIO, VAL_RATIO, SEED
        )
        splits = {"train": train_idx, "val": val_idx, "test": test_idx}

        for split, indices in splits.items():
            out_dir = dest / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in indices:
                src_path = paths[i]
                # Normalise to .jpg extension for consistency
                dst_name = src_path.stem + ".jpg"
                dst_path = out_dir / dst_name
                if src_path.suffix.lower() in {".tif", ".tiff"}:
                    with Image.open(src_path) as im:
                        im.convert("RGB").save(dst_path, "JPEG", quality=95)
                else:
                    shutil.copy2(src_path, dst_path)
                try:
                    with Image.open(dst_path) as im:
                        w, h = im.size
                except Exception:
                    w, h = 0, 0
                rows.append(
                    {
                        "image_id": image_id,
                        "file_name": f"{split}/{label}/{dst_name}",
                        "split": split,
                        "label": label,
                        "width": w,
                        "height": h,
                    }
                )
                image_id += 1
            print(f"  {split}: {len(indices)} images")

    csv_path = dest / "metadata.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "file_name", "split", "label", "width", "height"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote metadata.csv: {len(rows)} rows → {csv_path}")
    print(f"Done. Dataset written to: {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=str(DEFAULT_SRC))
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    args = parser.parse_args()
    restructure(Path(args.src), Path(args.dest))
