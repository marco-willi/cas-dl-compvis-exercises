"""Step A1 — Download and explore Snapshot Kgalagadi metadata.

Downloads the COCO Camera Traps JSON from LILA Science, parses categories,
sequences, images, and annotations, and prints per-category image counts
after one-per-sequence deduplication.

Writes selected_kgalagadi.json with chosen category IDs.
"""

import json
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METADATA_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "snapshot-safari/KGA/SnapshotKgalagadi_S1_v1.0.json.zip"
)
OUTPUT_DIR = Path(__file__).resolve().parent
MIN_IMAGES_PER_CLASS = 30  # minimum after deduplication to consider a class

# ---------------------------------------------------------------------------
# Download & extract
# ---------------------------------------------------------------------------
print(f"Downloading metadata from:\n  {METADATA_URL}")
resp = requests.get(METADATA_URL, timeout=120)
resp.raise_for_status()

with zipfile.ZipFile(BytesIO(resp.content)) as zf:
    json_names = [n for n in zf.namelist() if n.endswith(".json")]
    assert json_names, f"No JSON file found in zip. Contents: {zf.namelist()}"
    json_name = json_names[0]
    print(f"Extracting: {json_name}")
    data = json.loads(zf.read(json_name))

# ---------------------------------------------------------------------------
# Parse structure
# ---------------------------------------------------------------------------
print(f"\nTop-level keys: {list(data.keys())}")

categories = {c["id"]: c["name"] for c in data["categories"]}
print(f"Categories ({len(categories)}): {categories}")

images = {img["id"]: img for img in data["images"]}
print(f"Total images: {len(images)}")

annotations = data["annotations"]
print(f"Total annotations: {len(annotations)}")

# Check if there are sequence IDs on images
sample_img = data["images"][0]
print(f"\nSample image keys: {list(sample_img.keys())}")
has_seq = "seq_id" in sample_img or "sequence_id" in sample_img
seq_key = (
    "seq_id"
    if "seq_id" in sample_img
    else "sequence_id"
    if "sequence_id" in sample_img
    else None
)
print(f"Sequence key on images: {seq_key}")

if seq_key:
    sequences = {img.get(seq_key) for img in data["images"]}
    print(f"Unique sequences: {len(sequences)}")

# ---------------------------------------------------------------------------
# Map annotations → images
# ---------------------------------------------------------------------------
# Build image_id → category_id mapping
img_to_cat = {}
for ann in annotations:
    img_id = ann["image_id"]
    cat_id = ann["category_id"]
    img_to_cat[img_id] = cat_id

# Count all images per category
all_counts = Counter()
for _img_id, cat_id in img_to_cat.items():
    all_counts[categories.get(cat_id, f"unknown_{cat_id}")] += 1

print("\n" + "=" * 60)
print("ALL IMAGES PER CATEGORY (before deduplication)")
print("=" * 60)
print(f"{'category':<30} {'count':>8}")
print("-" * 40)
for name, count in all_counts.most_common():
    print(f"{name:<30} {count:>8}")
print(f"{'TOTAL':<30} {sum(all_counts.values()):>8}")

# ---------------------------------------------------------------------------
# One-per-sequence deduplication
# ---------------------------------------------------------------------------
if seq_key:
    # Group images by (sequence, category), keep first image per sequence
    seq_cat_first = {}  # (seq_id, cat_id) -> first image
    for img in data["images"]:
        img_id = img["id"]
        if img_id not in img_to_cat:
            continue
        cat_id = img_to_cat[img_id]
        s = img.get(seq_key)
        key = (s, cat_id)
        if key not in seq_cat_first:
            seq_cat_first[key] = img

    dedup_counts = Counter()
    for (_, cat_id), _ in seq_cat_first.items():
        dedup_counts[categories.get(cat_id, f"unknown_{cat_id}")] += 1

    print("\n" + "=" * 60)
    print("IMAGES PER CATEGORY (after one-per-sequence dedup)")
    print("=" * 60)
    print(f"{'category':<30} {'all':>8} {'dedup':>8} {'ratio':>8}")
    print("-" * 56)
    for name, count in all_counts.most_common():
        d = dedup_counts.get(name, 0)
        ratio = f"{d / count:.1%}" if count > 0 else "n/a"
        print(f"{name:<30} {count:>8} {d:>8} {ratio:>8}")
    print(
        f"{'TOTAL':<30} {sum(all_counts.values()):>8} {sum(dedup_counts.values()):>8}"
    )
else:
    dedup_counts = all_counts
    print("\nNo sequence key found — skipping deduplication.")

# ---------------------------------------------------------------------------
# Select classes
# ---------------------------------------------------------------------------
EXCLUDE_CLASSES = {"human"}  # exclude human, include empty

selected = []
for name, count in dedup_counts.most_common():
    if name.lower() in EXCLUDE_CLASSES:
        continue
    if count >= MIN_IMAGES_PER_CLASS:
        selected.append(
            {
                "name": name,
                "category_id": next(k for k, v in categories.items() if v == name),
                "dedup_count": count,
                "all_count": all_counts[name],
            }
        )

print("\n" + "=" * 60)
print(
    f"SELECTED CLASSES (>= {MIN_IMAGES_PER_CLASS} images after dedup, excluding {EXCLUDE_CLASSES})"
)
print("=" * 60)
for s in selected:
    print(
        f"  {s['name']:<25} id={s['category_id']:<5} dedup={s['dedup_count']:<6} all={s['all_count']}"
    )

print(f"\nTotal selected classes: {len(selected)}")
print(f"Total selected images (dedup): {sum(s['dedup_count'] for s in selected)}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = OUTPUT_DIR / "selected_kgalagadi.json"
with out_path.open("w") as f:
    json.dump(
        {
            "source": "Snapshot Kgalagadi S1 v1.0",
            "url": METADATA_URL,
            "min_images_per_class": MIN_IMAGES_PER_CLASS,
            "selected": selected,
        },
        f,
        indent=2,
    )
print(f"\nSaved selection to: {out_path}")

# ---------------------------------------------------------------------------
# Also dump image base URL for downstream steps
# ---------------------------------------------------------------------------
print("\nImage base URL (Azure):")
print(
    "  https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshot-safari/KGA/KGA_public"
)
print("\nDone.")
