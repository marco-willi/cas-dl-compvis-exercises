"""Step 2 — Explore SER species and MegaDetector detection coverage.

Loads the SER metadata and MD results produced by step 01. Maps sequence-level
labels to images, cross-tabulates species against MD animal detections, and
writes selected_ser.json with the chosen category IDs.

Writes:
    selected_ser.json — chosen categories with expected image counts

Usage:
    python scripts/prepare_datasets/ser/02_explore_species.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".cache"

SER_METADATA_PATH = CACHE_DIR / "ser_metadata.json"
MD_PATH = CACHE_DIR / "ser_md_results.json"

# Classes to exclude unconditionally (humans, vehicles, other non-wildlife)
EXCLUDE_CLASSES = {"human", "vehicle", "fire", "unidentifiable"}

# Target: aim for 5-8 species.  All with >= MIN_MD_IMAGES confident detections
# are eligible; we then pick the top N by MD coverage.
MIN_MD_IMAGES = 50  # lower bound — include iconic species even if rarer
MAX_CLASSES = 8
EMPTY_CLASS_SAMPLES = 50  # how many empty images to include
MD_CONF_THRESHOLD = 0.8
MD_EMPTY_CONF_MAX = 0.2  # max animal conf to qualify as "empty"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading SER metadata...")
with SER_METADATA_PATH.open() as f:
    meta = json.load(f)

print("Loading MD results...")
with MD_PATH.open() as f:
    md_data = json.load(f)

categories = {c["id"]: c["name"] for c in meta["categories"]}
images = {img["id"]: img for img in meta["images"]}
annotations = meta["annotations"]

print(f"  Images:      {len(images):,}")
print(f"  Annotations: {len(annotations):,}")
print(f"  Categories:  {len(categories):,}")

# ---------------------------------------------------------------------------
# Detect annotation structure (image_id vs seq_id)
# ---------------------------------------------------------------------------
sample_ann = annotations[0] if annotations else {}
has_image_id = "image_id" in sample_ann
seq_key_ann = (
    "seq_id"
    if "seq_id" in sample_ann
    else "sequence_id"
    if "sequence_id" in sample_ann
    else None
)

sample_img = next(iter(images.values()))
img_seq_key = (
    "seq_id"
    if "seq_id" in sample_img
    else "sequence_id"
    if "sequence_id" in sample_img
    else None
)
print(f"\n  Annotation key: {'image_id' if has_image_id else seq_key_ann}")
print(f"  Image seq key:  {img_seq_key}")

# ---------------------------------------------------------------------------
# Map labels to images
# ---------------------------------------------------------------------------
# Build image_id -> category_id
img_to_cat: dict[str, int] = {}

if has_image_id:
    # Per-image annotations
    for ann in annotations:
        img_to_cat[ann["image_id"]] = ann["category_id"]
elif seq_key_ann and img_seq_key:
    # Sequence-level annotations: map seq_id -> category_id
    seq_to_cat: dict[str, int] = {}
    for ann in annotations:
        seq_id = ann.get(seq_key_ann)
        if seq_id is not None:
            seq_to_cat[seq_id] = ann["category_id"]
    # Propagate to all images in that sequence
    for img_id, img in images.items():
        seq_id = img.get(img_seq_key)
        if seq_id in seq_to_cat:
            img_to_cat[img_id] = seq_to_cat[seq_id]
else:
    raise SystemExit("Cannot map labels to images: unknown annotation structure.")

print(f"  Images with label: {len(img_to_cat):,}")

# ---------------------------------------------------------------------------
# Build MD lookup: file_name -> best animal detection
# ---------------------------------------------------------------------------
# MD results use relative file paths; match by stripping common prefix if needed
md_by_file: dict[str, dict] = {}
for md_img in md_data.get("images", []):
    fname = md_img.get("file", md_img.get("file_name", ""))
    # Normalise: strip leading "./" or dataset prefix if present
    fname = fname.lstrip("./")
    md_by_file[fname] = md_img

# Also index by just the basename for fallback
md_by_basename: dict[str, dict] = {}
for fname, md_img in md_by_file.items():
    md_by_basename[Path(fname).name] = md_img


def get_md_entry(img: dict) -> dict | None:
    """Return the MD entry for an image, trying multiple key strategies."""
    fname = img.get("file_name", "")
    if fname in md_by_file:
        return md_by_file[fname]
    # Try stripping dataset prefix
    stripped = fname.lstrip("./")
    if stripped in md_by_file:
        return md_by_file[stripped]
    # Try basename only
    return md_by_basename.get(Path(fname).name)


def best_animal_detection(md_entry: dict | None) -> tuple[float, list | None]:
    """Return (best_conf, bbox) for the highest-conf animal detection, or (0, None)."""
    if md_entry is None:
        return 0.0, None
    best_conf = 0.0
    best_bbox = None
    for det in md_entry.get("detections", []):
        if str(det.get("category", "")) == "1":  # animal
            conf = float(det.get("conf", 0))
            if conf > best_conf:
                best_conf = conf
                best_bbox = det.get("bbox")
    return best_conf, best_bbox


# Check MD coverage for a few sample images
sample_imgs = list(images.values())[:5]
print("\nSample MD lookup check:")
for img in sample_imgs:
    md_entry = get_md_entry(img)
    conf, _ = best_animal_detection(md_entry)
    print(
        f"  {img['file_name'][:60]:60s}  md_found={md_entry is not None}  conf={conf:.2f}"
    )

# ---------------------------------------------------------------------------
# Per-species stats: total images and MD-covered images
# ---------------------------------------------------------------------------
all_counts: Counter = Counter()
md_counts: Counter = Counter()
md_conf_sum: dict[str, float] = defaultdict(float)

for img_id, img in images.items():
    cat_id = img_to_cat.get(img_id)
    if cat_id is None:
        continue
    cat_name = categories.get(cat_id, f"unknown_{cat_id}")
    all_counts[cat_name] += 1

    md_entry = get_md_entry(img)
    conf, _ = best_animal_detection(md_entry)
    if conf >= MD_CONF_THRESHOLD:
        md_counts[cat_name] += 1
        md_conf_sum[cat_name] += conf

print("\n" + "=" * 75)
print(
    f"TOP-30 SPECIES: total images vs MD animal detections (conf ≥ {MD_CONF_THRESHOLD:.0%})"
)
print("=" * 75)
print(f"{'species':<28} {'total':>8} {'md_det':>8} {'coverage':>10} {'avg_conf':>10}")
print("-" * 70)
for name, total in all_counts.most_common(30):
    md = md_counts.get(name, 0)
    cov = f"{md / total:.0%}" if total > 0 else "n/a"
    avg_c = f"{md_conf_sum[name] / md:.3f}" if md > 0 else "n/a"
    print(f"{name:<28} {total:>8,} {md:>8,} {cov:>10} {avg_c:>10}")

# ---------------------------------------------------------------------------
# Empty class: images labelled "empty" where MD also finds nothing
# ---------------------------------------------------------------------------
empty_cat_name = None
for _cat_id, cat_name in categories.items():
    if cat_name.lower() in {"empty", "blank", "nothing"}:
        empty_cat_name = cat_name
        break

empty_candidates = []
if empty_cat_name:
    for img_id, img in images.items():
        cat_id = img_to_cat.get(img_id)
        if categories.get(cat_id, "").lower() == empty_cat_name.lower():
            md_entry = get_md_entry(img)
            conf, _ = best_animal_detection(md_entry)
            if conf < MD_EMPTY_CONF_MAX:
                seq_id = img.get(img_seq_key, img_id) if img_seq_key else img_id
                empty_candidates.append({"img": img, "seq_id": seq_id})

print(f"\nEmpty class ('{empty_cat_name}'): {len(empty_candidates):,} candidate images")
print(f"  (MD animal conf < {MD_EMPTY_CONF_MAX:.0%})")

# ---------------------------------------------------------------------------
# Select species
# ---------------------------------------------------------------------------
eligible = [
    (name, md_counts[name], all_counts[name])
    for name, total in all_counts.most_common()
    if name.lower() not in EXCLUDE_CLASSES
    and name.lower() != (empty_cat_name or "").lower()
    and md_counts.get(name, 0) >= MIN_MD_IMAGES
]

# Sort by MD coverage count, take top MAX_CLASSES
eligible.sort(key=lambda x: x[1], reverse=True)
chosen = eligible[:MAX_CLASSES]

print("\n" + "=" * 60)
print(
    f"SELECTED SPECIES (top {MAX_CLASSES} by MD coverage, ≥{MIN_MD_IMAGES} MD images)"
)
print("=" * 60)
selected = []
for name, md_cnt, total_cnt in chosen:
    cat_id = next(k for k, v in categories.items() if v == name)
    print(f"  {name:<25}  id={cat_id:<6}  md_det={md_cnt:<6,}  total={total_cnt:,}")
    selected.append(
        {
            "name": name,
            "category_id": cat_id,
            "md_count": md_cnt,
            "total_count": total_cnt,
        }
    )

# Add empty class entry
if empty_cat_name and empty_candidates:
    empty_cat_id = next(
        k for k, v in categories.items() if v.lower() == empty_cat_name.lower()
    )
    selected.append(
        {
            "name": "empty",
            "category_id": empty_cat_id,
            "md_count": 0,
            "total_count": len(empty_candidates),
            "note": f"MD conf < {MD_EMPTY_CONF_MAX}; {EMPTY_CLASS_SAMPLES} will be sampled",
        }
    )
    print(f"  {'empty':<25}  id={empty_cat_id:<6}  (MD-confirmed empty frames)")

print(f"\nTotal selected classes: {len(selected)}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = SCRIPT_DIR / "selected_ser.json"
with out_path.open("w") as f:
    json.dump(
        {
            "source": "Snapshot Safari 2024 Expansion — SER (Serengeti)",
            "md_conf_threshold": MD_CONF_THRESHOLD,
            "md_empty_conf_max": MD_EMPTY_CONF_MAX,
            "min_md_images": MIN_MD_IMAGES,
            "max_classes": MAX_CLASSES,
            "empty_class_samples": EMPTY_CLASS_SAMPLES,
            "selected": selected,
        },
        f,
        indent=2,
    )
print(f"\nSaved selection to: {out_path}")
print(
    "\nDone. Review the table above and edit selected_ser.json if needed before running step 03."
)
