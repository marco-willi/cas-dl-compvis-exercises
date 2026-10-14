"""Step 2 — Filter, sample, and build manifest for ABO furniture subset.

Reads furniture_items.json from Step 1, resolves image paths from the cached
images.csv.gz, builds train/val/test splits by item_id, and writes the
manifest and retrieval groups.

Usage:
    python scripts/prepare_datasets/abo/02_filter_sample.py
"""

import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".cache"
SEED = 42
MAX_VIEWS = 4  # 1 primary + up to 3 additional views
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
MIN_ITEMS_PER_CLASS = 100

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
print("Loading furniture items from Step 1...")
with (SCRIPT_DIR / "furniture_items.json").open() as f:
    items = json.load(f)
print(f"  Loaded {len(items)} items")

print("Loading image metadata...")
image_meta = {}
with gzip.open(CACHE_DIR / "images.csv.gz", "rt", encoding="utf-8") as gz:
    reader = csv.DictReader(gz)
    for row in reader:
        image_meta[row["image_id"]] = {
            "height": int(row["height"]),
            "width": int(row["width"]),
            "path": row["path"],
        }
print(f"  Loaded {len(image_meta)} image records")

# ---------------------------------------------------------------------------
# Build per-item image lists with resolved paths
# ---------------------------------------------------------------------------
print("\nResolving image paths...")
skipped_no_path = 0
valid_items = []

for item in items:
    all_image_ids = []
    if item["main_image_id"]:
        all_image_ids.append(item["main_image_id"])
    for img_id in item.get("other_image_ids", []):
        if img_id not in all_image_ids:  # deduplicate
            all_image_ids.append(img_id)

    # Resolve paths and filter to images with metadata
    resolved = []
    for img_id in all_image_ids:
        if img_id in image_meta:
            resolved.append(
                {
                    "image_id": img_id,
                    "path": image_meta[img_id]["path"],
                    "height": image_meta[img_id]["height"],
                    "width": image_meta[img_id]["width"],
                }
            )
        else:
            skipped_no_path += 1

    if not resolved:
        continue

    # Limit to MAX_VIEWS images per item
    resolved = resolved[:MAX_VIEWS]

    valid_items.append(
        {
            "item_id": item["item_id"],
            "label": item["coarse_category"],
            "item_name": item["item_name"],
            "brand": item["brand"],
            "color": item["color"],
            "material": item["material"],
            "images": resolved,
        }
    )

# Deduplicate by item_id (same product can appear across domains/product_types)
seen_ids = set()
deduped = []
for item in valid_items:
    if item["item_id"] not in seen_ids:
        seen_ids.add(item["item_id"])
        deduped.append(item)
if len(deduped) < len(valid_items):
    print(
        f"  Deduplicated: {len(valid_items)} → {len(deduped)} (removed {len(valid_items) - len(deduped)} cross-domain duplicates)"
    )
valid_items = deduped

print(f"  Valid items (with resolved images): {len(valid_items)}")
print(f"  Skipped image IDs (no metadata): {skipped_no_path}")

# ---------------------------------------------------------------------------
# Drop categories below minimum
# ---------------------------------------------------------------------------
cat_counts = Counter(item["label"] for item in valid_items)
dropped = {cat for cat, count in cat_counts.items() if count < MIN_ITEMS_PER_CLASS}
if dropped:
    print(f"\n  Dropping categories below {MIN_ITEMS_PER_CLASS} items: {dropped}")
    valid_items = [item for item in valid_items if item["label"] not in dropped]

# ---------------------------------------------------------------------------
# Stratified split by item_id
# ---------------------------------------------------------------------------
print("\nBuilding stratified splits...")
rng = random.Random(SEED)

# Group items by category
by_category = defaultdict(list)
for item in valid_items:
    by_category[item["label"]].append(item)

split_items = {"train": [], "val": [], "test": []}

for cat in sorted(by_category):
    cat_items = by_category[cat]
    rng.shuffle(cat_items)

    n = len(cat_items)
    n_val = max(1, round(n * SPLIT_RATIOS["val"]))
    n_test = max(1, round(n * SPLIT_RATIOS["test"]))
    n_train = n - n_val - n_test

    split_items["train"].extend(cat_items[:n_train])
    split_items["val"].extend(cat_items[n_train : n_train + n_val])
    split_items["test"].extend(cat_items[n_train + n_val :])

# ---------------------------------------------------------------------------
# Build manifest (one entry per image, globally deduplicated)
# ---------------------------------------------------------------------------
# Many product variants share the same image_id (e.g. a generic product shot
# reused across color variants). We assign each image_id to exactly one entry
# — the first item that claims it — to keep the manifest in 1-to-1 correspondence
# with the files on disk.
print("Building manifest...")
manifest = []
retrieval_groups = {}
seen_image_ids: set[str] = set()
shared_image_ids_skipped = 0

for split_name, s_items in split_items.items():
    for item in s_items:
        group_image_ids = []
        view_idx = 0
        for img in item["images"]:
            img_id = img["image_id"]
            if img_id in seen_image_ids:
                shared_image_ids_skipped += 1
                continue
            seen_image_ids.add(img_id)
            entry = {
                "image_id": img_id,
                "item_id": item["item_id"],
                "s3_path": f"images/small/{img['path']}",
                "label": item["label"],
                "split": split_name,
                "is_primary": view_idx == 0,
                "view_index": view_idx,
                "original_height": img["height"],
                "original_width": img["width"],
            }
            manifest.append(entry)
            group_image_ids.append(img_id)
            view_idx += 1

        retrieval_groups[item["item_id"]] = {
            "label": item["label"],
            "split": split_name,
            "image_ids": group_image_ids,
        }

if shared_image_ids_skipped:
    print(
        f"  Skipped {shared_image_ids_skipped} duplicate image_ids shared across items"
    )

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
print("\nValidating...")

# Check no item_id spans multiple splits
item_splits = defaultdict(set)
for entry in manifest:
    item_splits[entry["item_id"]].add(entry["split"])
leaks = {iid for iid, splits in item_splits.items() if len(splits) > 1}
assert not leaks, f"Item IDs spanning multiple splits: {leaks}"
print("  No item leaks across splits")

# Check primary image counts — items whose primary image_id was already claimed
# by an earlier item will have their next unique image promoted to view_index=0.
primary_count = sum(1 for e in manifest if e["is_primary"])
items_with_images = sum(1 for g in retrieval_groups.values() if g["image_ids"])
items_without_images = len(valid_items) - items_with_images
if items_without_images:
    print(
        f"  WARNING: {items_without_images} items have no images after dedup (all image_ids claimed)"
    )
print(
    f"  Primary images: {primary_count} (items with at least one unique image: {items_with_images})"
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MANIFEST SUMMARY")
print("=" * 70)

print(f"\n  {'split':<8} {'items':>8} {'images':>8} {'primary':>10}")
print("  " + "-" * 38)
for split_name in ["train", "val", "test"]:
    s_entries = [e for e in manifest if e["split"] == split_name]
    s_items_count = len({e["item_id"] for e in s_entries})
    s_primary = sum(1 for e in s_entries if e["is_primary"])
    print(f"  {split_name:<8} {s_items_count:>8} {len(s_entries):>8} {s_primary:>10}")
total_items = len({e["item_id"] for e in manifest})
print(f"  {'TOTAL':<8} {total_items:>8} {len(manifest):>8} {primary_count:>10}")

print(f"\n  {'category':<12} {'train':>8} {'val':>8} {'test':>8} {'total':>8}")
print("  " + "-" * 48)
categories = sorted({e["label"] for e in manifest})
for cat in categories:
    counts = {}
    for split_name in ["train", "val", "test"]:
        counts[split_name] = len(
            {
                e["item_id"]
                for e in manifest
                if e["label"] == cat and e["split"] == split_name
            }
        )
    total = sum(counts.values())
    print(
        f"  {cat:<12} {counts['train']:>8} {counts['val']:>8} {counts['test']:>8} {total:>8}"
    )

print(f"\n  Retrieval groups: {len(retrieval_groups)}")
avg_group_size = sum(len(g["image_ids"]) for g in retrieval_groups.values()) / len(
    retrieval_groups
)
print(f"  Avg images per group: {avg_group_size:.1f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
manifest_path = SCRIPT_DIR / "abo_manifest.json"
with manifest_path.open("w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nSaved manifest ({len(manifest)} entries) to: {manifest_path}")

groups_path = SCRIPT_DIR / "abo_retrieval_groups.json"
with groups_path.open("w") as f:
    json.dump(retrieval_groups, f, indent=2)
print(f"Saved retrieval groups ({len(retrieval_groups)} groups) to: {groups_path}")

print("\nDone.")
