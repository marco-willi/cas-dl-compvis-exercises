"""Step 2 — Filter, sample, and build manifest for DeepFashion classroom subset.

Reads selected_deepfashion.json (Step 1) and list_eval_partition.txt, then:
  1. Applies source→coarse label mapping.
  2. Keeps only items with >= MIN_IMAGES_PER_ITEM images.
  3. Samples up to MAX_ITEMS_PER_CLASS items per coarse class (seed 42).
  4. Stratified 70/15/15 train/val/test split by item_id.
  5. Writes deepfashion_manifest.json.

Usage:
    python scripts/prepare_datasets/deepfashion/02_filter_subset.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "deepfashion"
PARTITION_FILE = RAW_DIR / "Eval" / "list_eval_partition.txt"

SEED = 42
MAX_ITEMS_PER_CLASS = 100
MIN_IMAGES_PER_ITEM = 2  # need query + at least one gallery match
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ---------------------------------------------------------------------------
# Load selected_deepfashion.json from Step 1
# ---------------------------------------------------------------------------
selection_path = SCRIPT_DIR / "selected_deepfashion.json"
if not selection_path.exists():
    raise FileNotFoundError(
        f"selected_deepfashion.json not found: {selection_path}\n"
        "Run 01_explore_metadata.py first."
    )

with selection_path.open() as f:
    selection = json.load(f)

source_to_coarse: dict[str, str] = selection["source_to_coarse"]
selected_labels: set[str] = {s["coarse_label"] for s in selection["selected"]}

print("Loaded selection:")
for s in selection["selected"]:
    print(
        f"  {s['coarse_label']:<15} items={s['item_count']} images={s['image_count']}"
    )

# ---------------------------------------------------------------------------
# Parse list_eval_partition.txt
# ---------------------------------------------------------------------------
print(f"\nParsing {PARTITION_FILE}...")
if not PARTITION_FILE.exists():
    raise FileNotFoundError(
        f"Partition file not found: {PARTITION_FILE}\n"
        "Ensure DeepFashion raw data is at data/raw/deepfashion/."
    )

records: list[dict] = []
with PARTITION_FILE.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3 or not parts[0].startswith("img/"):
            continue
        path_str, item_id, benchmark_split = parts
        path_parts = Path(path_str).parts
        if len(path_parts) < 4:
            continue
        source_category = path_parts[2]
        coarse = source_to_coarse.get(source_category)
        if coarse not in selected_labels:
            continue
        records.append(
            {
                "image_path": path_str,
                "item_id": item_id,
                "source_category": source_category,
                "coarse_label": coarse,
            }
        )

print(f"  Records matching selected coarse labels: {len(records)}")

# ---------------------------------------------------------------------------
# Group images by item_id; keep only items with enough images
# ---------------------------------------------------------------------------
item_images: dict[str, list[str]] = defaultdict(list)
item_label: dict[str, str] = {}

for rec in records:
    item_id = rec["item_id"]
    item_images[item_id].append(rec["image_path"])
    item_label[item_id] = rec["coarse_label"]

# Filter to items with at least MIN_IMAGES_PER_ITEM images
valid_items = {
    iid: paths
    for iid, paths in item_images.items()
    if len(paths) >= MIN_IMAGES_PER_ITEM
}

print(f"  Items with >= {MIN_IMAGES_PER_ITEM} images: {len(valid_items)}")
dropped = len(item_images) - len(valid_items)
if dropped:
    print(f"  Dropped {dropped} items with < {MIN_IMAGES_PER_ITEM} images")

# ---------------------------------------------------------------------------
# Per-class item counts after filtering
# ---------------------------------------------------------------------------
by_class: dict[str, list[str]] = defaultdict(list)
for iid in valid_items:
    by_class[item_label[iid]].append(iid)

print("\nPer-class item counts after filtering:")
print(f"  {'label':<15} {'items':>8}")
print("  " + "-" * 25)
for label in sorted(by_class):
    print(f"  {label:<15} {len(by_class[label]):>8}")

# ---------------------------------------------------------------------------
# Sample up to MAX_ITEMS_PER_CLASS per class
# ---------------------------------------------------------------------------
rng = random.Random(SEED)
sampled_by_class: dict[str, list[str]] = {}
for label in sorted(by_class):
    items_list = sorted(by_class[label])  # deterministic order before shuffle
    rng.shuffle(items_list)
    sampled_by_class[label] = items_list[:MAX_ITEMS_PER_CLASS]

print(f"\nAfter sampling (max {MAX_ITEMS_PER_CLASS} items/class):")
print(f"  {'label':<15} {'items':>8}")
print("  " + "-" * 25)
for label in sorted(sampled_by_class):
    print(f"  {label:<15} {len(sampled_by_class[label]):>8}")
total_sampled_items = sum(len(v) for v in sampled_by_class.values())
print(f"\n  Total sampled items: {total_sampled_items}")

# ---------------------------------------------------------------------------
# Stratified train/val/test split by item_id
# ---------------------------------------------------------------------------
print("\nBuilding stratified splits by item_id...")
split_assignment: dict[str, str] = {}

for label in sorted(sampled_by_class):
    items_list = list(sampled_by_class[label])  # already shuffled above
    n = len(items_list)
    n_val = max(1, round(n * SPLIT_RATIOS["val"]))
    n_test = max(1, round(n * SPLIT_RATIOS["test"]))
    n_train = n - n_val - n_test

    for iid in items_list[:n_train]:
        split_assignment[iid] = "train"
    for iid in items_list[n_train : n_train + n_val]:
        split_assignment[iid] = "val"
    for iid in items_list[n_train + n_val :]:
        split_assignment[iid] = "test"

# ---------------------------------------------------------------------------
# Build manifest
# ---------------------------------------------------------------------------
print("Building manifest...")
manifest: list[dict] = []

for iid, paths in sorted(valid_items.items()):
    if iid not in split_assignment:
        continue
    split = split_assignment[iid]
    label = item_label[iid]
    for idx, image_path in enumerate(sorted(paths)):
        frame = Path(image_path).stem  # e.g. "01_1_front"
        image_id = f"{iid}_{frame}"
        manifest.append(
            {
                "image_id": image_id,
                "item_id": iid,
                "source_path": image_path,
                "label": label,
                "split": split,
                "frame_index": idx,
                "width": None,  # populated in Step 3 after copy
                "height": None,
            }
        )

print(f"  Total manifest entries: {len(manifest)}")

# ---------------------------------------------------------------------------
# Validation — no item_id spans multiple splits
# ---------------------------------------------------------------------------
item_splits: dict[str, set] = defaultdict(set)
for entry in manifest:
    item_splits[entry["item_id"]].add(entry["split"])
leaks = {iid for iid, splits in item_splits.items() if len(splits) > 1}
assert not leaks, f"Item IDs spanning multiple splits: {leaks}"
print("  Validation: no item leaks across splits")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MANIFEST SUMMARY")
print("=" * 70)

print(f"\n  {'split':<8} {'items':>8} {'images':>8}")
print("  " + "-" * 26)
for split in ["train", "val", "test"]:
    s_entries = [e for e in manifest if e["split"] == split]
    s_items = len({e["item_id"] for e in s_entries})
    print(f"  {split:<8} {s_items:>8} {len(s_entries):>8}")
total_items = len({e["item_id"] for e in manifest})
print(f"  {'TOTAL':<8} {total_items:>8} {len(manifest):>8}")

print(f"\n  {'label':<15} {'train':>8} {'val':>8} {'test':>8} {'total':>8}")
print("  " + "-" * 49)
labels = sorted({e["label"] for e in manifest})
for label in labels:
    counts = {
        split: len(
            {
                e["item_id"]
                for e in manifest
                if e["label"] == label and e["split"] == split
            }
        )
        for split in ["train", "val", "test"]
    }
    total = sum(counts.values())
    print(
        f"  {label:<15} {counts['train']:>8} {counts['val']:>8} {counts['test']:>8} {total:>8}"
    )

# Class balance check
item_counts_by_label = Counter(item_label[iid] for iid in split_assignment)
max_count = max(item_counts_by_label.values())
min_count = min(item_counts_by_label.values())
ratio = max_count / min_count
print(f"\n  Class balance (items): max={max_count} min={min_count} ratio={ratio:.1f}x")
if ratio > 3:
    print(
        "  WARNING: class imbalance > 3x — consider adjusting MAX_ITEMS_PER_CLASS or merging small classes"
    )
else:
    print("  OK: class balance within acceptable range")

# ---------------------------------------------------------------------------
# Save manifest
# ---------------------------------------------------------------------------
manifest_path = SCRIPT_DIR / "deepfashion_manifest.json"
with manifest_path.open("w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nSaved manifest ({len(manifest)} entries) to: {manifest_path}")
print("\nDone. Run 03_copy_images.py next.")
