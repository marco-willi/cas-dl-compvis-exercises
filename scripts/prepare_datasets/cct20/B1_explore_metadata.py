"""Step B1 — Download and explore Caltech Camera Traps metadata.

Downloads two files:
1. The full Caltech Camera Traps COCO JSON (image-level annotations with categories)
2. The CCT20 benchmark annotation tar.gz (train/val splits, image lists)

Parses both, cross-references to get per-category counts for CCT20 images,
and prints a ranked table. Writes selected_cct20.json with chosen category IDs.
"""

import json
import tarfile
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Full dataset annotations (has category info per image)
FULL_META_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltechcameratraps/labels/caltech_camera_traps.json.zip"
)

# CCT20 benchmark annotations (has train/val splits and image lists)
CCT20_META_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltechcameratraps/eccv_18_annotations.tar.gz"
)

OUTPUT_DIR = Path(__file__).resolve().parent
MIN_IMAGES_PER_CLASS = 200  # minimum to consider a class for the subset
MAX_SELECTED_CLASSES = 8

# ---------------------------------------------------------------------------
# Download full metadata
# ---------------------------------------------------------------------------
print(f"Downloading full metadata from:\n  {FULL_META_URL}")
resp = requests.get(FULL_META_URL, timeout=120)
resp.raise_for_status()

with zipfile.ZipFile(BytesIO(resp.content)) as zf:
    json_names = [n for n in zf.namelist() if n.endswith(".json")]
    assert json_names, f"No JSON in zip. Contents: {zf.namelist()}"
    json_name = json_names[0]
    print(f"Extracting: {json_name}")
    full_data = json.loads(zf.read(json_name))

print(f"Top-level keys: {list(full_data.keys())}")

categories = {c["id"]: c["name"] for c in full_data["categories"]}
print(f"Categories ({len(categories)}):")
for cid, name in sorted(categories.items()):
    print(f"  {cid:>3}: {name}")

images_full = {img["id"]: img for img in full_data["images"]}
print(f"Total images in full dataset: {len(images_full)}")

# Build image_id → category_id from annotations
img_to_cat = {}
for ann in full_data["annotations"]:
    img_to_cat[ann["image_id"]] = ann["category_id"]

# ---------------------------------------------------------------------------
# Download CCT20 metadata
# ---------------------------------------------------------------------------
print(f"\nDownloading CCT20 annotations from:\n  {CCT20_META_URL}")
resp2 = requests.get(CCT20_META_URL, timeout=120)
resp2.raise_for_status()

cct20_jsons = {}
with tarfile.open(fileobj=BytesIO(resp2.content), mode="r:gz") as tf:
    print(f"CCT20 tar contents: {tf.getnames()}")
    for member in tf.getmembers():
        if member.name.endswith(".json"):
            key = Path(member.name).stem
            cct20_jsons[key] = json.loads(tf.extractfile(member).read())

print(f"CCT20 JSON files: {list(cct20_jsons.keys())}")

# ---------------------------------------------------------------------------
# Explore CCT20 structure
# ---------------------------------------------------------------------------
# Print structure of each CCT20 JSON
for name, js in cct20_jsons.items():
    if isinstance(js, dict):
        print(f"\n  {name}: keys={list(js.keys())}")
        for k, v in js.items():
            if isinstance(v, list):
                print(f"    {k}: list[{len(v)}]", end="")
                if v:
                    print(
                        f"  sample={v[0]}"
                        if not isinstance(v[0], dict)
                        else f"  sample_keys={list(v[0].keys())}"
                    )
                else:
                    print()
            elif isinstance(v, dict):
                print(f"    {k}: dict[{len(v)}]")
            else:
                print(f"    {k}: {type(v).__name__} = {v}")
    elif isinstance(js, list):
        print(f"\n  {name}: list[{len(js)}]")

# ---------------------------------------------------------------------------
# Collect CCT20 image IDs and cross-reference with full annotations
# ---------------------------------------------------------------------------
# Try to find image IDs from CCT20 JSONs
cct20_image_ids = set()
cct20_split_map = {}  # image_id -> split name

for split_name, js in cct20_jsons.items():
    if isinstance(js, dict) and "images" in js:
        for img in js["images"]:
            img_id = img["id"] if isinstance(img, dict) else img
            cct20_image_ids.add(img_id)
            cct20_split_map[img_id] = split_name
    elif isinstance(js, dict) and "annotations" in js:
        for ann in js["annotations"]:
            img_id = ann["image_id"]
            cct20_image_ids.add(img_id)
            if img_id not in cct20_split_map:
                cct20_split_map[img_id] = split_name

print(f"\nCCT20 unique image IDs: {len(cct20_image_ids)}")

# Cross-reference: how many CCT20 images have annotations in the full dataset?
cct20_annotated = {iid for iid in cct20_image_ids if iid in img_to_cat}
print(f"CCT20 images with full-dataset annotations: {len(cct20_annotated)}")
print(f"CCT20 images without annotations: {len(cct20_image_ids - cct20_annotated)}")

# ---------------------------------------------------------------------------
# Per-category counts for CCT20 images
# ---------------------------------------------------------------------------
cct20_counts = Counter()
for img_id in cct20_annotated:
    cat_id = img_to_cat[img_id]
    cct20_counts[categories.get(cat_id, f"unknown_{cat_id}")] += 1

print("\n" + "=" * 60)
print("CCT20 IMAGES PER CATEGORY")
print("=" * 60)
print(f"{'category':<30} {'count':>8}")
print("-" * 40)
for name, count in cct20_counts.most_common():
    print(f"{name:<30} {count:>8}")
print(f"{'TOTAL':<30} {sum(cct20_counts.values()):>8}")

# Also show full dataset counts for comparison
full_counts = Counter()
for cat_id in img_to_cat.values():
    full_counts[categories.get(cat_id, f"unknown_{cat_id}")] += 1

print("\n" + "=" * 60)
print("FULL DATASET IMAGES PER CATEGORY (for reference)")
print("=" * 60)
print(f"{'category':<30} {'full':>8} {'cct20':>8}")
print("-" * 50)
for name, count in full_counts.most_common():
    c20 = cct20_counts.get(name, 0)
    print(f"{name:<30} {count:>8} {c20:>8}")

# ---------------------------------------------------------------------------
# Per-split counts
# ---------------------------------------------------------------------------
split_counts = Counter(cct20_split_map.values())
print(f"\nCCT20 splits: {dict(split_counts)}")

# ---------------------------------------------------------------------------
# Select classes
# ---------------------------------------------------------------------------
selected = []
for name, count in cct20_counts.most_common():
    if count >= MIN_IMAGES_PER_CLASS and len(selected) < MAX_SELECTED_CLASSES:
        cat_id = next(k for k, v in categories.items() if v == name)
        selected.append(
            {
                "name": name,
                "category_id": cat_id,
                "cct20_count": count,
                "full_count": full_counts[name],
            }
        )

print("\n" + "=" * 60)
print(
    f"SELECTED CLASSES (top {MAX_SELECTED_CLASSES} with >= {MIN_IMAGES_PER_CLASS} CCT20 images, including empty)"
)
print("=" * 60)
for s in selected:
    print(
        f"  {s['name']:<25} id={s['category_id']:<5} cct20={s['cct20_count']:<6} full={s['full_count']}"
    )

print(f"\nTotal selected classes: {len(selected)}")
print(f"Total selected images (CCT20): {sum(s['cct20_count'] for s in selected)}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = OUTPUT_DIR / "selected_cct20.json"
result = {
    "source": "Caltech Camera Traps — CCT20 benchmark subset",
    "full_meta_url": FULL_META_URL,
    "cct20_meta_url": CCT20_META_URL,
    "min_images_per_class": MIN_IMAGES_PER_CLASS,
    "max_selected_classes": MAX_SELECTED_CLASSES,
    "selected": selected,
}

with out_path.open("w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved selection to: {out_path}")

# ---------------------------------------------------------------------------
# Image base URLs
# ---------------------------------------------------------------------------
print("\nImage base URLs:")
print(
    "  Azure: https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images"
)
print(
    "  GCP (CCT20 images tar): https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz"
)
print("\nDone.")
