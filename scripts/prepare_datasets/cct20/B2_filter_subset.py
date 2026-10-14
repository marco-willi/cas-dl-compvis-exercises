"""Step B2 — Filter and sample CCT20 subset.

Reads the full Caltech Camera Traps metadata + CCT20 annotations,
cross-references to get per-image labels for CCT20 images, samples
up to MAX_PER_CLASS images per selected category with a stratified
80/10/10 train/val/test split, and writes cct20_manifest.json.
"""

import json
import random
import tarfile
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FULL_META_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltechcameratraps/labels/caltech_camera_traps.json.zip"
)
CCT20_META_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltechcameratraps/eccv_18_annotations.tar.gz"
)
IMAGE_BASE_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "caltech-unzipped/cct_images"
)
SCRIPT_DIR = Path(__file__).resolve().parent
SELECTION_PATH = SCRIPT_DIR / "selected_cct20.json"
SEED = 42
MAX_PER_CLASS = 200
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.80, 0.10, 0.10

# ---------------------------------------------------------------------------
# Load selection from B1
# ---------------------------------------------------------------------------
with SELECTION_PATH.open() as f:
    selection = json.load(f)

selected_cat_ids = {s["category_id"] for s in selection["selected"]}
cat_id_to_name = {s["category_id"]: s["name"] for s in selection["selected"]}
print(f"Selected categories: {cat_id_to_name}")

# ---------------------------------------------------------------------------
# Download full metadata (for category labels per image)
# ---------------------------------------------------------------------------
print(f"Downloading full metadata from:\n  {FULL_META_URL}")
resp = requests.get(FULL_META_URL, timeout=120)
resp.raise_for_status()

with zipfile.ZipFile(BytesIO(resp.content)) as zf:
    json_name = next(n for n in zf.namelist() if n.endswith(".json"))
    full_data = json.loads(zf.read(json_name))

categories = {c["id"]: c["name"] for c in full_data["categories"]}
full_images = {img["id"]: img for img in full_data["images"]}

# image_id -> category_id
img_to_cat = {}
for ann in full_data["annotations"]:
    img_to_cat[ann["image_id"]] = ann["category_id"]

# ---------------------------------------------------------------------------
# Download CCT20 annotations (for image lists)
# ---------------------------------------------------------------------------
print(f"Downloading CCT20 annotations from:\n  {CCT20_META_URL}")
resp2 = requests.get(CCT20_META_URL, timeout=120)
resp2.raise_for_status()

cct20_images = {}  # image_id -> image dict (from CCT20 JSONs, has file_name)
with tarfile.open(fileobj=BytesIO(resp2.content), mode="r:gz") as tf:
    for member in tf.getmembers():
        if not member.name.endswith(".json"):
            continue
        js = json.loads(tf.extractfile(member).read())
        if "images" in js:
            for img in js["images"]:
                cct20_images[img["id"]] = img

print(f"CCT20 total images: {len(cct20_images)}")

# ---------------------------------------------------------------------------
# Cross-reference: CCT20 images with selected categories
# ---------------------------------------------------------------------------
by_label = defaultdict(list)
for img_id, img in cct20_images.items():
    if img_id not in img_to_cat:
        continue
    cat_id = img_to_cat[img_id]
    if cat_id not in selected_cat_ids:
        continue
    label = cat_id_to_name[cat_id]
    by_label[label].append(
        {
            "image_id": img_id,
            "file_name": img["file_name"],
            "category_id": cat_id,
            "label": label,
            "width": img.get("width", 0),
            "height": img.get("height", 0),
            "location": img.get("location", ""),
            "seq_id": img.get("seq_id", ""),
            "datetime": img.get("date_captured", ""),
        }
    )

print("\nAvailable images per selected class:")
for label in sorted(by_label):
    print(f"  {label:<25} {len(by_label[label]):>6}")

# ---------------------------------------------------------------------------
# Sample up to MAX_PER_CLASS per category, stratified split
# ---------------------------------------------------------------------------
random.seed(SEED)
manifest = []

for label in sorted(by_label):
    entries = by_label[label]
    random.shuffle(entries)
    sampled = entries[:MAX_PER_CLASS]

    n = len(sampled)
    n_train = max(1, round(n * TRAIN_RATIO))
    n_val = max(1, round(n * VAL_RATIO))

    for i, entry in enumerate(sampled):
        if i < n_train:
            entry["split"] = "train"
        elif i < n_train + n_val:
            entry["split"] = "val"
        else:
            entry["split"] = "test"
        entry["url"] = f"{IMAGE_BASE_URL}/{entry['file_name']}"
        manifest.append(entry)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
split_label_counts = defaultdict(Counter)
for entry in manifest:
    split_label_counts[entry["split"]][entry["label"]] += 1

print("\n" + "=" * 70)
print("MANIFEST SUMMARY")
print("=" * 70)
print(f"{'label':<25} {'train':>8} {'val':>8} {'test':>8} {'total':>8}")
print("-" * 60)
all_labels = sorted({e["label"] for e in manifest})
for label in all_labels:
    tr = split_label_counts["train"][label]
    va = split_label_counts["val"][label]
    te = split_label_counts["test"][label]
    print(f"{label:<25} {tr:>8} {va:>8} {te:>8} {tr + va + te:>8}")

totals = {s: sum(c.values()) for s, c in split_label_counts.items()}
print(
    f"{'TOTAL':<25} {totals.get('train', 0):>8} {totals.get('val', 0):>8} {totals.get('test', 0):>8} {sum(totals.values()):>8}"
)

# Check for sequence leakage
seq_splits = defaultdict(set)
for entry in manifest:
    if entry["seq_id"]:
        seq_splits[entry["seq_id"]].add(entry["split"])
leaked = {s: splits for s, splits in seq_splits.items() if len(splits) > 1}
if leaked:
    print(
        f"\nNote: {len(leaked)} sequences span multiple splits (acceptable for CCT20 — sampling is per-image, not per-sequence)."
    )
else:
    print("\nNo sequence leakage across splits.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = SCRIPT_DIR / "cct20_manifest.json"
with out_path.open("w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nSaved manifest ({len(manifest)} entries) to: {out_path}")
