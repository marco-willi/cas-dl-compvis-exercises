"""Step 3 — Copy and organise DeepFashion images into ImageFolder layout.

Reads deepfashion_manifest.json (Step 2), copies each source image from
data/raw/deepfashion/img/... to:

    data/deepfashion/<split>/<label>/<item_id>_<frame>.jpg

Records actual width and height per copied image, then writes:

    data/deepfashion/metadata.csv

Usage:
    python scripts/prepare_datasets/deepfashion/03_copy_images.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "deepfashion" / "img"
OUTPUT_DIR = REPO_ROOT / "data" / "deepfashion"
MANIFEST_PATH = SCRIPT_DIR / "deepfashion_manifest.json"
METADATA_PATH = OUTPUT_DIR / "metadata.csv"
FAILED_LOG_PATH = OUTPUT_DIR / "failed_copies.txt"

# ---------------------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------------------
if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}\nRun 02_filter_subset.py first."
    )

with MANIFEST_PATH.open() as f:
    manifest: list[dict] = json.load(f)

print(f"Manifest: {MANIFEST_PATH}")
print(f"Entries:  {len(manifest)}")
print(f"Source:   {RAW_DIR}")
print(f"Output:   {OUTPUT_DIR}\n")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Copy images
# ---------------------------------------------------------------------------
metadata_rows: list[dict] = []
failed: list[tuple[str, str]] = []
skipped = 0
copied = 0

for idx, entry in enumerate(manifest, start=1):
    image_id = entry["image_id"]
    item_id = entry["item_id"]
    split = entry["split"]
    label = entry["label"]

    # Source path is relative to RAW_DIR's parent (the raw deepfashion root),
    # e.g. "img/MEN/Dresses/id_00000001/01_1_front.jpg"
    # RAW_DIR points to .../raw/deepfashion/img, so strip the "img/" prefix.
    source_rel = entry["source_path"]  # "img/MEN/..."
    source_rel_no_prefix = Path(source_rel).relative_to("img")
    source_path = RAW_DIR / source_rel_no_prefix

    dest_dir = OUTPUT_DIR / split / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_filename = f"{image_id}.jpg"
    dest_path = dest_dir / dest_filename

    if dest_path.exists():
        # Read existing image dimensions for metadata
        try:
            with Image.open(dest_path) as im:
                width, height = im.size
        except Exception:
            width, height = None, None
        metadata_rows.append(
            {
                "image_id": image_id,
                "item_id": item_id,
                "file_name": f"{split}/{label}/{dest_filename}",
                "split": split,
                "label": label,
                "width": width,
                "height": height,
            }
        )
        skipped += 1
        continue

    if not source_path.exists():
        failed.append((image_id, f"source not found: {source_path}"))
        continue

    try:
        with Image.open(source_path) as im:
            im = im.convert("RGB")
            width, height = im.size
            im.save(dest_path, format="JPEG", quality=92)
        copied += 1
    except (UnidentifiedImageError, OSError) as exc:
        failed.append((image_id, str(exc)))
        continue

    metadata_rows.append(
        {
            "image_id": image_id,
            "item_id": item_id,
            "file_name": f"{split}/{label}/{dest_filename}",
            "split": split,
            "label": label,
            "width": width,
            "height": height,
        }
    )

    if idx % 500 == 0 or idx == len(manifest):
        print(
            f"[{idx}/{len(manifest)}] copied={copied} skipped={skipped} failed={len(failed)}"
        )

# ---------------------------------------------------------------------------
# Write metadata.csv
# ---------------------------------------------------------------------------
fieldnames = ["image_id", "item_id", "file_name", "split", "label", "width", "height"]
with METADATA_PATH.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(metadata_rows)

print(f"\nWrote metadata.csv: {METADATA_PATH} ({len(metadata_rows)} rows)")

# ---------------------------------------------------------------------------
# Log failures
# ---------------------------------------------------------------------------
if failed:
    with FAILED_LOG_PATH.open("w") as f:
        for image_id, reason in failed:
            f.write(f"{image_id}\t{reason}\n")
    print(f"Failed copies: {len(failed)} — see {FAILED_LOG_PATH}")
else:
    print("Failed copies: 0")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COPY SUMMARY")
print("=" * 60)
print(f"  Copied:  {copied}")
print(f"  Skipped: {skipped} (already existed)")
print(f"  Failed:  {len(failed)}")
print(f"  Metadata rows: {len(metadata_rows)}")
assert len(metadata_rows) == len(manifest) - len(failed), (
    f"Row count mismatch: {len(metadata_rows)} metadata rows vs "
    f"{len(manifest) - len(failed)} expected"
)

# Split / label breakdown
from collections import Counter  # noqa: E402 (after early-exit checks above)

split_label_counts: Counter = Counter(
    (row["split"], row["label"]) for row in metadata_rows
)
splits = ["train", "val", "test"]
labels = sorted({row["label"] for row in metadata_rows})

print(f"\n  {'label':<15} {'train':>8} {'val':>8} {'test':>8} {'total':>8}")
print("  " + "-" * 49)
for label in labels:
    counts = [split_label_counts.get((s, label), 0) for s in splits]
    print(
        f"  {label:<15} {counts[0]:>8} {counts[1]:>8} {counts[2]:>8} {sum(counts):>8}"
    )

print(
    "\nDone. Verify with:"
    "\n  from torchvision.datasets import ImageFolder"
    "\n  ds = ImageFolder('data/deepfashion/train')"
    "\n  print(ds.classes, len(ds))"
)
print("Then run 04_quality_check.ipynb for visual inspection.")
