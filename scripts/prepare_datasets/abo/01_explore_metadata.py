"""Step 1 — Download ABO metadata and explore furniture categories.

Downloads the Amazon Berkeley Objects listings archive and image metadata,
parses product types, filters to furniture-related categories, and prints
per-category item/image counts.

Writes selected_abo_categories.json with chosen categories and counts.

Usage:
    python scripts/prepare_datasets/abo/01_explore_metadata.py
"""

import csv
import gzip
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LISTINGS_URL = (
    "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar"
)
IMAGES_META_URL = (
    "https://amazon-berkeley-objects.s3.amazonaws.com/images/metadata/images.csv.gz"
)

OUTPUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
MIN_ITEMS_PER_CLASS = 100

# Furniture-related keywords to match against product_type (case-insensitive).
# These are broad enough to catch ABO taxonomy variants.
FURNITURE_KEYWORDS = [
    "chair",
    "table",
    "desk",
    "sofa",
    "couch",
    "bed",
    "lamp",
    "light",
    "shelf",
    "shelving",
    "bookcase",
    "cabinet",
    "dresser",
    "nightstand",
    "bench",
    "stool",
    "ottoman",
    "futon",
    "mattress",
    "wardrobe",
    "chest",
    "vanity",
    "mirror",
    "rug",
    "curtain",
]

# Exact product_type → coarse label mapping.
# Only include product types that are genuinely furniture — no substring matching
# on noisy categories like PORTABLE_*, VEGETABLE, HOME_BED_AND_BATH, etc.
EXACT_MAPPING: dict[str, str] = {
    # Seating
    "SOFA": "sofa",
    "CHAIR": "chair",
    "OTTOMAN": "chair",
    "STOOL_SEATING": "chair",
    "BENCH": "chair",
    "BEAN_BAG_CHAIR": "chair",
    # Tables
    "TABLE": "table",
    "DESK": "table",
    # Beds
    "BED": "bed",
    "BED_FRAME": "bed",
    "HEADBOARD": "bed",
    "MATTRESS": "bed",
    # Lighting
    "LAMP": "lamp",
    "LIGHT_FIXTURE": "lamp",
    "HOME_LIGHTING_AND_LAMPS": "lamp",
    # Storage
    "SHELF": "storage",
    "CABINET": "storage",
    "DRESSER": "storage",
    "BOOKCASE": "storage",
}


def _to_coarse(product_type: str) -> str | None:
    """Map a raw product_type string to a coarse furniture category."""
    return EXACT_MAPPING.get(product_type.upper().strip())


def _extract_en(field: list[dict] | str | None) -> str:
    """Extract the English value from a multilingual ABO field."""
    if field is None:
        return ""
    if isinstance(field, str):
        return field
    for entry in field:
        if entry.get("language_tag", "").startswith("en"):
            return entry.get("value", "")
    # Fallback: return first value
    if field:
        return field[0].get("value", "")
    return ""


# ---------------------------------------------------------------------------
# Download & cache
# ---------------------------------------------------------------------------
CACHE_DIR.mkdir(parents=True, exist_ok=True)

listings_cache = CACHE_DIR / "abo-listings.tar"
images_meta_cache = CACHE_DIR / "images.csv.gz"


def _download(url: str, dest: Path) -> None:
    """Download a file with progress, skip if already cached."""
    if dest.exists():
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  [cached] {dest.name} ({size_mb:.1f} MB)")
        return
    print(f"  Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(
                    f"\r  {downloaded / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f} MB ({pct:.0f}%)",
                    end="",
                    flush=True,
                )
    print()


print("Downloading ABO metadata...")
_download(LISTINGS_URL, listings_cache)
_download(IMAGES_META_URL, images_meta_cache)

# ---------------------------------------------------------------------------
# Parse listings
# ---------------------------------------------------------------------------
print("\nParsing listings...")
items = []  # list of dicts with fields we care about

with tarfile.open(listings_cache, "r") as tar:
    members = sorted(
        [m for m in tar.getmembers() if m.name.endswith(".json.gz")],
        key=lambda m: m.name,
    )
    print(f"  Found {len(members)} listing files in archive")

    for member in members:
        f = tar.extractfile(member)
        if f is None:
            continue
        with gzip.open(f, "rt", encoding="utf-8") as gz:
            for line in gz:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                product_type = _extract_en(obj.get("product_type"))
                if not product_type:
                    continue

                # Collect image IDs
                image_ids = []
                main_img = obj.get("main_image_id")
                if main_img:
                    image_ids.append(main_img)
                other_imgs = obj.get("other_image_id", [])
                if other_imgs:
                    image_ids.extend(other_imgs)

                if not image_ids:
                    continue  # skip items with no images

                items.append(
                    {
                        "item_id": obj.get("item_id", ""),
                        "domain_name": obj.get("domain_name", ""),
                        "product_type": product_type,
                        "item_name": _extract_en(obj.get("item_name")),
                        "brand": _extract_en(obj.get("brand")),
                        "color": _extract_en(obj.get("color")),
                        "material": _extract_en(obj.get("material")),
                        "style": _extract_en(obj.get("style")),
                        "main_image_id": main_img,
                        "other_image_ids": other_imgs,
                        "num_images": len(image_ids),
                    }
                )

print(f"  Total items with product_type + images: {len(items)}")

# ---------------------------------------------------------------------------
# Parse image metadata
# ---------------------------------------------------------------------------
print("\nParsing image metadata...")
image_meta = {}  # image_id -> {height, width, path}

with gzip.open(images_meta_cache, "rt", encoding="utf-8") as gz:
    reader = csv.DictReader(gz)
    for row in reader:
        image_meta[row["image_id"]] = {
            "height": int(row["height"]),
            "width": int(row["width"]),
            "path": row["path"],
        }

print(f"  Total images in metadata: {len(image_meta)}")

# ---------------------------------------------------------------------------
# All product types — ranked
# ---------------------------------------------------------------------------
pt_counts = Counter(item["product_type"] for item in items)
print("\n" + "=" * 70)
print("TOP 50 PRODUCT TYPES (all items)")
print("=" * 70)
print(f"  {'product_type':<45} {'items':>8}")
print("  " + "-" * 55)
for pt, count in pt_counts.most_common(50):
    print(f"  {pt:<45} {count:>8}")
print(f"  {'...':<45}")
print(f"  {'TOTAL unique product_types':<45} {len(pt_counts):>8}")

# ---------------------------------------------------------------------------
# Filter to furniture
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FURNITURE FILTERING")
print("=" * 70)

furniture_items = []
for item in items:
    coarse = _to_coarse(item["product_type"])
    if coarse:
        item["coarse_category"] = coarse
        furniture_items.append(item)

print(f"  Furniture items matched: {len(furniture_items)}")

# Show raw product_type → coarse mapping
raw_to_coarse = defaultdict(lambda: Counter())
for item in furniture_items:
    raw_to_coarse[item["coarse_category"]][item["product_type"]] += 1

print("\n  Raw product_type → coarse category mapping:")
for coarse in sorted(raw_to_coarse):
    print(f"\n  [{coarse}]")
    for raw_pt, count in raw_to_coarse[coarse].most_common():
        print(f"    {raw_pt:<40} {count:>6}")

# ---------------------------------------------------------------------------
# Per-category counts
# ---------------------------------------------------------------------------
coarse_counts = Counter(item["coarse_category"] for item in furniture_items)
coarse_image_counts = Counter()
for item in furniture_items:
    coarse_image_counts[item["coarse_category"]] += item["num_images"]

print("\n" + "=" * 70)
print("COARSE CATEGORY SUMMARY")
print("=" * 70)
print(f"  {'category':<20} {'items':>8} {'images':>8} {'img/item':>10}")
print("  " + "-" * 48)
for cat, count in coarse_counts.most_common():
    n_img = coarse_image_counts[cat]
    ratio = n_img / count if count > 0 else 0
    marker = " ✓" if count >= MIN_ITEMS_PER_CLASS else " ✗"
    print(f"  {cat:<20} {count:>8} {n_img:>8} {ratio:>9.1f}{marker}")
print("  " + "-" * 48)
print(
    f"  {'TOTAL':<20} {sum(coarse_counts.values()):>8} {sum(coarse_image_counts.values()):>8}"
)

# ---------------------------------------------------------------------------
# Check image metadata coverage for furniture items
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("IMAGE METADATA COVERAGE (furniture items)")
print("=" * 70)

matched = 0
missing = 0
for item in furniture_items:
    all_ids = []
    if item["main_image_id"]:
        all_ids.append(item["main_image_id"])
    all_ids.extend(item["other_image_ids"])
    for img_id in all_ids:
        if img_id in image_meta:
            matched += 1
        else:
            missing += 1

print(f"  Images found in images.csv: {matched}")
print(f"  Images missing from images.csv: {missing}")
print(f"  Coverage: {matched / (matched + missing) * 100:.1f}%")

# ---------------------------------------------------------------------------
# Select categories
# ---------------------------------------------------------------------------
selected = []
for cat, count in coarse_counts.most_common():
    if count >= MIN_ITEMS_PER_CLASS:
        selected.append(
            {
                "coarse_category": cat,
                "item_count": count,
                "image_count": coarse_image_counts[cat],
                "raw_product_types": dict(raw_to_coarse[cat].most_common()),
            }
        )

print("\n" + "=" * 70)
print(f"SELECTED CATEGORIES (>= {MIN_ITEMS_PER_CLASS} items)")
print("=" * 70)
for s in selected:
    print(
        f"  {s['coarse_category']:<20} items={s['item_count']:<8} images={s['image_count']}"
    )

print(f"\n  Total selected categories: {len(selected)}")
print(f"  Total selected items: {sum(s['item_count'] for s in selected)}")
print(f"  Total selected images: {sum(s['image_count'] for s in selected)}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = OUTPUT_DIR / "selected_abo_categories.json"
result = {
    "source": "Amazon Berkeley Objects (ABO)",
    "source_url": LISTINGS_URL,
    "license": "CC BY-NC 4.0",
    "min_items_per_class": MIN_ITEMS_PER_CLASS,
    "total_items_parsed": len(items),
    "furniture_items_matched": len(furniture_items),
    "selected": selected,
}
with out_path.open("w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved selection to: {out_path}")

# Also save the full furniture items list for downstream steps
furniture_path = OUTPUT_DIR / "furniture_items.json"
with furniture_path.open("w") as f:
    json.dump(
        [
            item
            for item in furniture_items
            if item["coarse_category"] in {s["coarse_category"] for s in selected}
        ],
        f,
        indent=2,
    )
print(
    f"Saved {len([i for i in furniture_items if i['coarse_category'] in {s['coarse_category'] for s in selected}])} furniture items to: {furniture_path}"
)

print("\nDone.")
