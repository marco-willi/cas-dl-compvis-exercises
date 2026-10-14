"""Step A2 — Deduplicate and filter Kgalagadi subset.

Reads the full metadata JSON, applies one-per-sequence deduplication,
filters to selected classes from A1, creates a stratified train/val/test
split grouped by sequence ID, and writes kgalagadi_manifest.json.
"""

import json
import random
import zipfile
from collections import Counter, defaultdict
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
IMAGE_BASE_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "snapshot-safari/KGA/KGA_public"
)
SCRIPT_DIR = Path(__file__).resolve().parent
SELECTION_PATH = SCRIPT_DIR / "selected_kgalagadi.json"
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# ---------------------------------------------------------------------------
# Load selection from A1
# ---------------------------------------------------------------------------
with SELECTION_PATH.open() as f:
    selection = json.load(f)

selected_cat_ids = {s["category_id"] for s in selection["selected"]}
cat_id_to_name = {s["category_id"]: s["name"] for s in selection["selected"]}
print(f"Selected categories: {cat_id_to_name}")

# ---------------------------------------------------------------------------
# Download metadata
# ---------------------------------------------------------------------------
print(f"Downloading metadata from:\n  {METADATA_URL}")
resp = requests.get(METADATA_URL, timeout=120)
resp.raise_for_status()

with zipfile.ZipFile(BytesIO(resp.content)) as zf:
    json_name = next(n for n in zf.namelist() if n.endswith(".json"))
    data = json.loads(zf.read(json_name))

categories = {c["id"]: c["name"] for c in data["categories"]}
images = {img["id"]: img for img in data["images"]}

# Build image_id -> category_id
img_to_cat = {}
for ann in data["annotations"]:
    img_to_cat[ann["image_id"]] = ann["category_id"]

# ---------------------------------------------------------------------------
# One-per-sequence deduplication
# ---------------------------------------------------------------------------
# Group images by (seq_id, category_id), keep the first frame (frame_num=1
# or lowest frame_num) per sequence.
seq_groups = defaultdict(list)
for img in data["images"]:
    img_id = img["id"]
    if img_id not in img_to_cat:
        continue
    cat_id = img_to_cat[img_id]
    if cat_id not in selected_cat_ids:
        continue
    if img.get("corrupt", False):
        continue
    seq_groups[(img["seq_id"], cat_id)].append(img)

# Pick one image per sequence: the one with the lowest frame_num
deduped = []
for (seq_id, cat_id), imgs in seq_groups.items():
    imgs.sort(key=lambda x: x.get("frame_num", 0))
    best = imgs[0]
    deduped.append(
        {
            "image_id": best["id"],
            "file_name": best["file_name"],
            "seq_id": seq_id,
            "category_id": cat_id,
            "label": cat_id_to_name[cat_id],
            "width": best["width"],
            "height": best["height"],
            "location": best.get("location", ""),
            "datetime": best.get("datetime", ""),
        }
    )

print(f"Deduplicated images: {len(deduped)}")

# ---------------------------------------------------------------------------
# Stratified split by sequence ID (grouped so no sequence spans splits)
# ---------------------------------------------------------------------------
# Group entries by label
by_label = defaultdict(list)
for entry in deduped:
    by_label[entry["label"]].append(entry)

random.seed(SEED)
manifest = []

for _label, entries in sorted(by_label.items()):
    random.shuffle(entries)
    n = len(entries)
    n_train = max(1, round(n * TRAIN_RATIO))
    n_val = max(1, round(n * VAL_RATIO))
    # rest goes to test
    for i, entry in enumerate(entries):
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

# Verify no sequence spans multiple splits
seq_splits = defaultdict(set)
for entry in manifest:
    seq_splits[entry["seq_id"]].add(entry["split"])
leaked = {s: splits for s, splits in seq_splits.items() if len(splits) > 1}
if leaked:
    print(f"\nWARNING: {len(leaked)} sequences span multiple splits!")
else:
    print("\nNo sequence leakage across splits.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = SCRIPT_DIR / "kgalagadi_manifest.json"
with out_path.open("w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nSaved manifest ({len(manifest)} entries) to: {out_path}")
