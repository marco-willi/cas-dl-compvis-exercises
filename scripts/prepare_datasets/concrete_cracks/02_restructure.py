"""Step 02 — Restructure Concrete Cracks into ImageFolder layout with train/val/test splits.

Reads the raw extracted archive produced by 01_download.py. The source archive
(Özgenel et al., Mendeley) contains two flat class directories with no splits:

    Negative/  *.jpg   (~20,000 images, 227x227)
    Positive/  *.jpg   (~20,000 images, 227x227)

This script locates those class directories (handling variations in nesting depth),
applies a stratified 70/15/15 split, copies images into
data/concrete_cracks/<split>/<label>/, and writes a metadata.csv sidecar.

Output layout:
    data/concrete_cracks/
        train/
            Negative/  *.jpg
            Positive/  *.jpg
        val/
            Negative/  *.jpg
            Positive/  *.jpg
        test/
            Negative/  *.jpg
            Positive/  *.jpg
        metadata.csv
        README.md  (written by C2_package.py)

Usage:
    python scripts/prepare_datasets/concrete_cracks/02_restructure.py
    python scripts/prepare_datasets/concrete_cracks/02_restructure.py --src data/raw/concrete_cracks/extracted --dest data/concrete_cracks
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
EXPECTED_CLASSES = {"Negative", "Positive"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC = REPO_ROOT / "data" / "raw" / "concrete_cracks" / "extracted"
DEFAULT_DEST = REPO_ROOT / "data" / "concrete_cracks"


def find_class_dirs(root: Path) -> dict[str, Path]:
    """Walk root to find directories named Negative and Positive."""
    found = {}
    for p in root.rglob("*"):
        if p.is_dir() and p.name in EXPECTED_CLASSES:
            found[p.name] = p
    if len(found) < 2:
        raise RuntimeError(
            f"Could not find both class directories {EXPECTED_CLASSES} under {root}.\n"
            f"Found: {list(found.keys())}\n"
            "Check the layout with 01_download.py first."
        )
    return found


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
    assert src.exists(), f"Source dir not found: {src}\nRun 01_download.py first."

    if dest.exists():
        print(f"Destination already exists: {dest}")
        print("Remove it to re-run. Exiting.")
        return

    class_dirs = find_class_dirs(src)
    print(f"Found class directories: {list(class_dirs.keys())}")

    rows = []
    image_id = 0

    for label, class_dir in sorted(class_dirs.items()):
        print(f"\nProcessing class: {label} ({class_dir})")
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
