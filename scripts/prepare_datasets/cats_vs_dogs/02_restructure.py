"""Step 02 — Restructure Cats vs Dogs into ImageFolder layout with train/val/test splits.

Reads the raw PetImages/ directory produced by 01_download.py, applies a
stratified 70/15/15 split, copies images into data/cats_vs_dogs/<split>/<label>/,
and writes a metadata.csv sidecar.

Source layout (after 01_download.py):
    data/raw/cats_vs_dogs/extracted/PetImages/
        Cat/  *.jpg
        Dog/  *.jpg

Output layout:
    data/cats_vs_dogs/
        train/
            Cat/  *.jpg
            Dog/  *.jpg
        val/
            Cat/  *.jpg
            Dog/  *.jpg
        test/
            Cat/  *.jpg
            Dog/  *.jpg
        metadata.csv
        README.md  (written by C2_package.py)

Usage:
    python scripts/prepare_datasets/cats_vs_dogs/02_restructure.py
    python scripts/prepare_datasets/cats_vs_dogs/02_restructure.py --src data/raw/cats_vs_dogs/extracted/PetImages --dest data/cats_vs_dogs
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (remainder)
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC = REPO_ROOT / "data" / "raw" / "cats_vs_dogs" / "extracted" / "PetImages"
DEFAULT_DEST = REPO_ROOT / "data" / "cats_vs_dogs"


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
    assert src.exists(), (
        f"Source PetImages dir not found: {src}\nRun 01_download.py first."
    )

    if dest.exists():
        print(f"Destination already exists: {dest}")
        print("Remove it to re-run. Exiting.")
        return

    rows = []
    image_id = 0

    for label_dir in sorted(src.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        print(f"\nProcessing class: {label}")

        paths = iter_valid_images(label_dir)
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
                dst_path = out_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                try:
                    with Image.open(dst_path) as im:
                        w, h = im.size
                except Exception:
                    w, h = 0, 0
                rows.append(
                    {
                        "image_id": image_id,
                        "file_name": f"{split}/{label}/{src_path.name}",
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
