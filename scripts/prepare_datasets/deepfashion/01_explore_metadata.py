"""Step 1 — Verify access and explore DeepFashion In-Shop metadata.

Reads the official DeepFashion In-Shop Clothes Retrieval Benchmark files from
data/raw/deepfashion/ (manually downloaded via MMLAB/CUHK access request).

Parses:
  - Eval/list_eval_partition.txt  → image paths, item_ids, benchmark splits
  - img/ directory tree           → infers source clothing category from path

Prints per-category image and item counts, then writes selected_deepfashion.json
with the chosen coarse→source mapping.

Usage:
    python scripts/prepare_datasets/deepfashion/01_explore_metadata.py

Prerequisites:
    data/raw/deepfashion/img/                     (image tree)
    data/raw/deepfashion/Eval/list_eval_partition.txt
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "deepfashion"
PARTITION_FILE = RAW_DIR / "Eval" / "list_eval_partition.txt"
IMG_DIR = RAW_DIR / "img"
OUTPUT_DIR = SCRIPT_DIR

MIN_ITEMS_PER_CLASS = 50  # minimum items to consider a source category viable

# ---------------------------------------------------------------------------
# Coarse taxonomy mapping
# Source categories are inferred from the second path component under img/,
# e.g. img/MEN/Denim_Jackets_and_Coats/... → source = "Denim_Jackets_and_Coats"
#
# The mapping below consolidates ~50 fine-grained source categories into
# 6 coarse classes. Adjust after reviewing the counts from Step 1.
# ---------------------------------------------------------------------------
SOURCE_TO_COARSE: dict[str, str] = {
    # Tops (MEN + WOMEN)
    "Tees_Tanks": "tops",
    "Blouses_Shirts": "tops",
    "Graphic_Tees": "tops",
    "Sweatshirts_Hoodies": "tops",
    "Sweaters": "tops",
    "Cardigans": "tops",
    "Shirts_Polos": "tops",
    # Dresses / one-pieces
    "Dresses": "dresses",
    "Rompers_Jumpsuits": "dresses",
    # Outerwear
    "Jackets_Coats": "outerwear",
    "Jackets_Vests": "outerwear",
    "Suiting": "outerwear",
    # Pants / bottoms
    "Pants": "pants",
    "Denim": "pants",
    "Shorts": "pants",
    "Leggings": "pants",
    # Skirts
    "Skirts": "skirts",
}


# ---------------------------------------------------------------------------
# Step 1a — Verify raw files exist
# ---------------------------------------------------------------------------
print("Checking DeepFashion raw data...")

missing = []
if not RAW_DIR.exists():
    missing.append(str(RAW_DIR))
if not PARTITION_FILE.exists():
    missing.append(str(PARTITION_FILE))
if not IMG_DIR.exists():
    missing.append(str(IMG_DIR))

if missing:
    print("\nERROR: Missing required files/directories:")
    for m in missing:
        print(f"  {m}")
    print(
        "\nPlease complete the MMLAB/CUHK access request and download the DeepFashion"
        " In-Shop Clothes Retrieval Benchmark to data/raw/deepfashion/."
        "\nSee: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html"
    )
    sys.exit(1)

print(f"  RAW_DIR:        {RAW_DIR}")
print(f"  PARTITION_FILE: {PARTITION_FILE}")
print(f"  IMG_DIR:        {IMG_DIR}")

# ---------------------------------------------------------------------------
# Step 1b — Parse list_eval_partition.txt
# ---------------------------------------------------------------------------
# Format (header lines vary by release; the data lines look like):
#   img/MEN/Denim_Jackets_and_Coats/id_00000001/01_1_front.jpg  id_00000001  train
# The first line is the image count, the second is the column header.
# ---------------------------------------------------------------------------
print("\nParsing list_eval_partition.txt...")
records: list[dict] = []

with PARTITION_FILE.open() as f:
    lines = f.readlines()

# Skip header lines (non-data lines at the top)
data_lines = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    # Data lines have exactly 3 parts: path item_id split
    if len(parts) == 3 and parts[0].startswith("img/"):
        data_lines.append(parts)

print(f"  Parsed {len(data_lines)} image records")

for path_str, item_id, split in data_lines:
    path = Path(path_str)
    # Path structure: img/<GENDER>/<SOURCE_CATEGORY>/<ITEM_ID>/<FRAME>.jpg
    parts = path.parts
    if len(parts) < 4:
        continue
    gender = parts[1] if len(parts) > 1 else "UNKNOWN"
    source_category = parts[2] if len(parts) > 2 else "UNKNOWN"
    records.append(
        {
            "image_path": path_str,
            "item_id": item_id,
            "benchmark_split": split,  # train / query / gallery
            "gender": gender,
            "source_category": source_category,
        }
    )

print(f"  Records with parsed source_category: {len(records)}")

# ---------------------------------------------------------------------------
# Step 1c — Inventory: images and items per source category
# ---------------------------------------------------------------------------
cat_image_counts: Counter = Counter()
cat_item_counts: dict[str, set] = defaultdict(set)

for rec in records:
    cat = rec["source_category"]
    cat_image_counts[cat] += 1
    cat_item_counts[cat].add(rec["item_id"])

print("\n" + "=" * 70)
print("ALL SOURCE CATEGORIES (ranked by image count)")
print("=" * 70)
print(f"  {'source_category':<45} {'images':>8} {'items':>8} {'→ coarse':<15}")
print("  " + "-" * 80)
for cat, img_count in cat_image_counts.most_common():
    n_items = len(cat_item_counts[cat])
    coarse = SOURCE_TO_COARSE.get(cat, "—")
    print(f"  {cat:<45} {img_count:>8} {n_items:>8}   {coarse}")

print(f"\n  Total source categories: {len(cat_image_counts)}")
print(f"  Total records:           {len(records)}")

# ---------------------------------------------------------------------------
# Step 1d — Coarse category summary
# ---------------------------------------------------------------------------
coarse_image_counts: Counter = Counter()
coarse_item_counts: dict[str, set] = defaultdict(set)
coarse_source_map: dict[str, list[str]] = defaultdict(list)

for rec in records:
    cat = rec["source_category"]
    coarse = SOURCE_TO_COARSE.get(cat)
    if coarse:
        coarse_image_counts[coarse] += 1
        coarse_item_counts[coarse].add(rec["item_id"])
        if cat not in coarse_source_map[coarse]:
            coarse_source_map[coarse].append(cat)

print("\n" + "=" * 70)
print("COARSE CATEGORY SUMMARY")
print("=" * 70)
print(f"  {'coarse_label':<15} {'images':>8} {'items':>8} {'img/item':>10}  sources")
print("  " + "-" * 70)
for coarse in sorted(coarse_image_counts, key=lambda c: -coarse_image_counts[c]):
    n_img = coarse_image_counts[coarse]
    n_items = len(coarse_item_counts[coarse])
    ratio = n_img / n_items if n_items > 0 else 0
    sources = ", ".join(sorted(coarse_source_map[coarse]))
    print(f"  {coarse:<15} {n_img:>8} {n_items:>8} {ratio:>9.1f}  {sources}")

# ---------------------------------------------------------------------------
# Step 1e — Benchmark split breakdown
# ---------------------------------------------------------------------------
split_counts: Counter = Counter(r["benchmark_split"] for r in records)
print("\n" + "=" * 70)
print("BENCHMARK SPLIT BREAKDOWN")
print("=" * 70)
for split, count in split_counts.most_common():
    print(f"  {split:<10} {count:>8} images")

# ---------------------------------------------------------------------------
# Step 1f — Select coarse categories with enough items
# ---------------------------------------------------------------------------
selected = []
for coarse in sorted(coarse_image_counts):
    n_items = len(coarse_item_counts[coarse])
    if n_items >= MIN_ITEMS_PER_CLASS:
        selected.append(
            {
                "coarse_label": coarse,
                "image_count": coarse_image_counts[coarse],
                "item_count": n_items,
                "source_categories": sorted(coarse_source_map[coarse]),
            }
        )

print("\n" + "=" * 70)
print(f"SELECTED COARSE CATEGORIES (>= {MIN_ITEMS_PER_CLASS} items)")
print("=" * 70)
for s in selected:
    print(
        f"  {s['coarse_label']:<15} items={s['item_count']:<6} images={s['image_count']}"
    )
print(f"\n  Total selected: {len(selected)} categories")
print(f"  Total items:    {sum(s['item_count'] for s in selected)}")
print(f"  Total images:   {sum(s['image_count'] for s in selected)}")

# ---------------------------------------------------------------------------
# Step 1g — Write selected_deepfashion.json
# ---------------------------------------------------------------------------
out_path = OUTPUT_DIR / "selected_deepfashion.json"
result = {
    "source": "DeepFashion In-Shop Clothes Retrieval Benchmark",
    "source_url": "https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html",
    "license": "Non-commercial, no redistribution — internal use only",
    "min_items_per_class": MIN_ITEMS_PER_CLASS,
    "total_records_parsed": len(records),
    "source_to_coarse": SOURCE_TO_COARSE,
    "selected": selected,
}
with out_path.open("w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved selection to: {out_path}")
print("\nDone. Review selected_deepfashion.json, adjust SOURCE_TO_COARSE if needed,")
print("then run 02_filter_subset.py.")
