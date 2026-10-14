"""Step 3 — Filter, deduplicate and build the SER manifest.

For each selected species:
  - Collect all images with that label.
  - Deduplicate: one image per sequence, picking the frame with the highest
    MegaDetector animal confidence.
  - Filter: keep only images with MD animal conf >= threshold.
  - Sample up to MAX_PER_CLASS images (seed=42).

Empty class:
  - Collect images labelled "empty" where MD finds nothing (conf < 0.2).
  - Sample ~50.

Stratified 70/15/15 train/val/test split, grouped by sequence_id so no
sequence spans two splits.

Writes:
    ser_manifest.json — one entry per image with label, split, md_bbox, md_conf

Usage:
    python scripts/prepare_datasets/ser/03_filter_manifest.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".cache"

SER_METADATA_PATH = CACHE_DIR / "ser_metadata.json"
MD_PATH = CACHE_DIR / "ser_md_results.json"
SELECTION_PATH = SCRIPT_DIR / "selected_ser.json"

IMAGE_BASE_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "snapshot-safari-2024-expansion"
)

MAX_PER_CLASS = 200
EMPTY_CLASS_SAMPLES = 50
MD_CONF_THRESHOLD = 0.8
MD_EMPTY_CONF_MAX = 0.2
SEED = 42
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15  # remainder goes to test

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading SER metadata...")
with SER_METADATA_PATH.open() as f:
    meta = json.load(f)

print("Loading MD results...")
with MD_PATH.open() as f:
    md_data = json.load(f)

print("Loading selection...")
with SELECTION_PATH.open() as f:
    selection = json.load(f)

categories = {c["id"]: c["name"] for c in meta["categories"]}
images = {img["id"]: img for img in meta["images"]}
annotations = meta["annotations"]

selected_entries = selection["selected"]
selected_non_empty = [s for s in selected_entries if s["name"] != "empty"]
empty_entry = next((s for s in selected_entries if s["name"] == "empty"), None)

print(f"  Selected animal classes: {[s['name'] for s in selected_non_empty]}")
print(f"  Empty class: {empty_entry is not None}")

# ---------------------------------------------------------------------------
# Detect annotation structure
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

# Build image_id -> category_id
img_to_cat: dict[str, int] = {}
if has_image_id:
    for ann in annotations:
        img_to_cat[ann["image_id"]] = ann["category_id"]
elif seq_key_ann and img_seq_key:
    seq_to_cat: dict[str, int] = {}
    for ann in annotations:
        seq_id = ann.get(seq_key_ann)
        if seq_id is not None:
            seq_to_cat[seq_id] = ann["category_id"]
    for img_id, img in images.items():
        seq_id = img.get(img_seq_key)
        if seq_id in seq_to_cat:
            img_to_cat[img_id] = seq_to_cat[seq_id]
else:
    raise SystemExit("Cannot map labels to images — unknown annotation structure.")

# ---------------------------------------------------------------------------
# MD lookup helpers
# ---------------------------------------------------------------------------
md_by_file: dict[str, dict] = {}
for md_img in md_data.get("images", []):
    fname = md_img.get("file", md_img.get("file_name", "")).lstrip("./")
    md_by_file[fname] = md_img

md_by_basename: dict[str, dict] = {}
for fname, md_img in md_by_file.items():
    md_by_basename[Path(fname).name] = md_img


def get_md_entry(img: dict) -> dict | None:
    fname = img.get("file_name", "").lstrip("./")
    if fname in md_by_file:
        return md_by_file[fname]
    return md_by_basename.get(Path(fname).name)


def best_animal_det(md_entry: dict | None) -> tuple[float, list | None]:
    if md_entry is None:
        return 0.0, None
    best_conf, best_bbox = 0.0, None
    for det in md_entry.get("detections", []):
        if str(det.get("category", "")) == "1":
            conf = float(det.get("conf", 0))
            if conf > best_conf:
                best_conf = conf
                best_bbox = det.get("bbox")
    return best_conf, best_bbox


# ---------------------------------------------------------------------------
# Helper: build manifest entries for one category
# ---------------------------------------------------------------------------
def build_entries_for_class(cat_id: int, label: str) -> list[dict]:
    """Collect, deduplicate and MD-filter images for one animal class."""
    # Group all images of this category by sequence
    seq_groups: dict[str, list[dict]] = defaultdict(list)
    for img_id, img in images.items():
        if img_to_cat.get(img_id) != cat_id:
            continue
        seq_id = img.get(img_seq_key, img_id) if img_seq_key else img_id
        md_entry = get_md_entry(img)
        conf, bbox = best_animal_det(md_entry)
        seq_groups[seq_id].append(
            {
                "img": img,
                "conf": conf,
                "bbox": bbox,
                "seq_id": seq_id,
            }
        )

    # One image per sequence: pick highest MD conf
    deduped = []
    for _seq_id, frames in seq_groups.items():
        best = max(frames, key=lambda x: x["conf"])
        if best["conf"] >= MD_CONF_THRESHOLD:
            deduped.append(best)

    return deduped


def make_manifest_entry(item: dict, label: str) -> dict:
    """Convert an internal dict to a manifest entry."""
    img = item["img"]
    file_name = img.get("file_name", "")
    url = f"{IMAGE_BASE_URL}/{file_name}"
    return {
        "image_id": img["id"],
        "file_name": file_name,
        "url": url,
        "label": label,
        "sequence_id": item["seq_id"],
        "split": "",  # filled later
        "md_bbox": item["bbox"],
        "md_conf": round(item["conf"], 4),
    }


# ---------------------------------------------------------------------------
# Collect entries per class
# ---------------------------------------------------------------------------
random.seed(SEED)
by_class: dict[str, list[dict]] = {}

print("\nCollecting and filtering per class...")
for sel in selected_non_empty:
    cat_id = sel["category_id"]
    label = sel["name"]
    entries = build_entries_for_class(cat_id, label)
    print(f"  {label:<25}: {len(entries):4} after dedup+MD filter", end="")
    if len(entries) > MAX_PER_CLASS:
        entries = random.sample(entries, MAX_PER_CLASS)
        print(f"  → sampled {MAX_PER_CLASS}", end="")
    print()
    by_class[label] = entries

# Empty class
if empty_entry:
    empty_cat_id = empty_entry["category_id"]
    empty_imgs = []
    for img_id, img in images.items():
        if img_to_cat.get(img_id) != empty_cat_id:
            continue
        md_entry = get_md_entry(img)
        conf, _ = best_animal_det(md_entry)
        if conf < MD_EMPTY_CONF_MAX:
            seq_id = img.get(img_seq_key, img_id) if img_seq_key else img_id
            empty_imgs.append({"img": img, "conf": 0.0, "bbox": None, "seq_id": seq_id})

    # One per sequence for empty too
    seq_seen: set[str] = set()
    deduped_empty = []
    for item in empty_imgs:
        if item["seq_id"] not in seq_seen:
            seq_seen.add(item["seq_id"])
            deduped_empty.append(item)

    print(f"  {'empty':<25}: {len(deduped_empty):4} after one-per-sequence", end="")
    if len(deduped_empty) > EMPTY_CLASS_SAMPLES:
        deduped_empty = random.sample(deduped_empty, EMPTY_CLASS_SAMPLES)
        print(f"  → sampled {EMPTY_CLASS_SAMPLES}", end="")
    print()
    by_class["empty"] = deduped_empty

# ---------------------------------------------------------------------------
# Stratified split by sequence_id (no sequence crosses split boundaries)
# ---------------------------------------------------------------------------
manifest: list[dict] = []

for label, entries in sorted(by_class.items()):
    # Shuffle sequence IDs then assign splits
    seq_ids = list({e["seq_id"] for e in entries})
    random.shuffle(seq_ids)
    n = len(seq_ids)
    n_train = max(1, round(n * TRAIN_RATIO))
    n_val = max(1, round(n * VAL_RATIO))

    seq_split: dict[str, str] = {}
    for i, seq_id in enumerate(seq_ids):
        if i < n_train:
            seq_split[seq_id] = "train"
        elif i < n_train + n_val:
            seq_split[seq_id] = "val"
        else:
            seq_split[seq_id] = "test"

    for item in entries:
        entry = make_manifest_entry(item, label)
        entry["split"] = seq_split[item["seq_id"]]
        manifest.append(entry)

# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------
print("\nValidation checks:")

# No sequence spans two splits
seq_to_splits: dict[str, set[str]] = defaultdict(set)
for entry in manifest:
    seq_to_splits[entry["sequence_id"]].add(entry["split"])
leaked = {s: splits for s, splits in seq_to_splits.items() if len(splits) > 1}
if leaked:
    print(f"  WARNING: {len(leaked)} sequences span multiple splits!")
else:
    print("  OK: No sequence leakage.")

# All animal entries have md_conf >= threshold
bad_conf = [
    e for e in manifest if e["label"] != "empty" and e["md_conf"] < MD_CONF_THRESHOLD
]
if bad_conf:
    print(
        f"  WARNING: {len(bad_conf)} animal entries with md_conf < {MD_CONF_THRESHOLD}"
    )
else:
    print(f"  OK: All animal entries have md_conf >= {MD_CONF_THRESHOLD}.")

# Empty entries have null bbox
bad_empty = [e for e in manifest if e["label"] == "empty" and e["md_bbox"] is not None]
if bad_empty:
    print(f"  WARNING: {len(bad_empty)} empty entries have non-null md_bbox")
else:
    print("  OK: Empty entries have null md_bbox.")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
split_label: dict[str, Counter] = defaultdict(Counter)
for entry in manifest:
    split_label[entry["split"]][entry["label"]] += 1

all_labels = sorted({e["label"] for e in manifest})
print("\n" + "=" * 75)
print("MANIFEST SUMMARY")
print("=" * 75)
print(f"{'label':<25} {'train':>8} {'val':>8} {'test':>8} {'total':>8}")
print("-" * 60)
for label in all_labels:
    tr = split_label["train"][label]
    va = split_label["val"][label]
    te = split_label["test"][label]
    print(f"{label:<25} {tr:>8} {va:>8} {te:>8} {tr + va + te:>8}")

totals = {s: sum(c.values()) for s, c in split_label.items()}
grand = sum(totals.values())
print(
    f"{'TOTAL':<25} {totals.get('train', 0):>8} {totals.get('val', 0):>8} "
    f"{totals.get('test', 0):>8} {grand:>8}"
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = SCRIPT_DIR / "ser_manifest.json"
with out_path.open("w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nSaved manifest ({len(manifest)} entries) to: {out_path}")
print("Done.")
