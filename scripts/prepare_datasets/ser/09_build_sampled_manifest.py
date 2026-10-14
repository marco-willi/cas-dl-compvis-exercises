"""Step 9 — Build ser_sampled manifest (5 000 images, realistic distribution).

Applies the same dedup + MD filter as step 3, but WITHOUT a hard per-class cap
of 200. Instead:
  - Each animal class is capped at MAX_PER_CLASS_SAMPLED = 1 000.
  - A global random sample of TOTAL_ANIMAL_SAMPLES = 4 950 is drawn from the
    combined per-class pools (proportional to pool sizes).
  - 50 empty images are added → 5 000 total.

This preserves the realistic Serengeti species distribution (wildebeest and
zebra dominate) while preventing any single class from crowding out all others.

Writes:
    scripts/prepare_datasets/ser/ser_sampled_manifest.json

Usage:
    python scripts/prepare_datasets/ser/09_build_sampled_manifest.py
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

TOTAL_ANIMAL_SAMPLES = 4_950  # animal images in final manifest (5000 total with empty)
EMPTY_CLASS_SAMPLES = 50  # empty images in final manifest
MD_CONF_THRESHOLD = 0.8
MD_EMPTY_CONF_MAX = 0.2
SEED = 42
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading SER metadata...")
with SER_METADATA_PATH.open() as f:
    meta = json.load(f)

print("Loading MD results...")
with MD_PATH.open() as f:
    md_data = json.load(f)

print("Loading species selection...")
with SELECTION_PATH.open() as f:
    selection = json.load(f)

categories = {c["id"]: c["name"] for c in meta["categories"]}
images = {img["id"]: img for img in meta["images"]}
annotations = meta["annotations"]

selected_entries = selection["selected"]
selected_non_empty = [s for s in selected_entries if s["name"] != "empty"]
empty_entry = next((s for s in selected_entries if s["name"] == "empty"), None)

print(f"  Selected animal classes: {[s['name'] for s in selected_non_empty]}")
print(f"  Total images in corpus:  {len(images):,}")

# ---------------------------------------------------------------------------
# Detect annotation structure (same logic as step 3)
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
img_to_cat: dict = {}
if has_image_id:
    for ann in annotations:
        img_to_cat[ann["image_id"]] = ann["category_id"]
elif seq_key_ann and img_seq_key:
    seq_to_cat: dict = {}
    for ann in annotations:
        sid = ann.get(seq_key_ann)
        if sid is not None:
            seq_to_cat[sid] = ann["category_id"]
    for img_id, img in images.items():
        sid = img.get(img_seq_key)
        if sid in seq_to_cat:
            img_to_cat[img_id] = seq_to_cat[sid]
else:
    raise SystemExit("Cannot map labels to images — unknown annotation structure.")

# ---------------------------------------------------------------------------
# MD lookup helpers
# ---------------------------------------------------------------------------
md_by_file: dict = {}
for md_img in md_data.get("images", []):
    fname = md_img.get("file", md_img.get("file_name", "")).lstrip("./")
    md_by_file[fname] = md_img

md_by_basename: dict = {}
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
# Build eligible pool per class (dedup + MD filter, cap at 1 000)
# ---------------------------------------------------------------------------
def build_entries_for_class(cat_id: int, label: str) -> list[dict]:
    seq_groups: dict = defaultdict(list)
    for img_id, img in images.items():
        if img_to_cat.get(img_id) != cat_id:
            continue
        seq_id = img.get(img_seq_key, img_id) if img_seq_key else img_id
        md_entry = get_md_entry(img)
        conf, bbox = best_animal_det(md_entry)
        seq_groups[seq_id].append(
            {"img": img, "conf": conf, "bbox": bbox, "seq_id": seq_id}
        )

    deduped = []
    for frames in seq_groups.values():
        best = max(frames, key=lambda x: x["conf"])
        if best["conf"] >= MD_CONF_THRESHOLD:
            deduped.append(best)
    return deduped


def make_manifest_entry(item: dict, label: str) -> dict:
    img = item["img"]
    file_name = img.get("file_name", "")
    return {
        "image_id": img["id"],
        "file_name": file_name,
        "url": f"{IMAGE_BASE_URL}/{file_name}",
        "label": label,
        "sequence_id": item["seq_id"],
        "split": "",
        "md_bbox": item["bbox"],
        "md_conf": round(item["conf"], 4),
    }


random.seed(SEED)

print("\nBuilding per-class pools (dedup + MD filter, NO CAP)...")
per_class_pools: dict[str, list[dict]] = {}

for sel in selected_non_empty:
    cat_id = sel["category_id"]
    label = sel["name"]
    entries = build_entries_for_class(cat_id, label)
    print(f"  {label:<25}: {len(entries):6} eligible")
    per_class_pools[label] = entries

# ---------------------------------------------------------------------------
# Proportional allocation from eligible pools
# ---------------------------------------------------------------------------
total_eligible = sum(len(pool) for pool in per_class_pools.values())
print(f"\nTotal eligible animal images: {total_eligible:,}")

# Compute allocations proportionally
allocations: dict[str, int] = {}
sampled_animal = []

for label in sorted(per_class_pools):
    pool = per_class_pools[label]
    fraction = len(pool) / total_eligible
    allocated = round(TOTAL_ANIMAL_SAMPLES * fraction)
    allocations[label] = allocated

    # Sample from the pool
    sample = random.sample(pool, min(allocated, len(pool)))
    for item in sample:
        sampled_animal.append((label, item))

print(f"\nProportional allocation (total {len(sampled_animal)}):")
for label in sorted(allocations):
    allocated = allocations[label]
    pct = 100 * allocated / TOTAL_ANIMAL_SAMPLES
    print(f"  {label:<25}: {allocated:>6}  ({pct:>5.1f}%)")

# ---------------------------------------------------------------------------
# Empty class (same as step 3)
# ---------------------------------------------------------------------------
print("\nBuilding empty class pool...")
empty_imgs = []
if empty_entry:
    empty_cat_id = empty_entry["category_id"]
    seq_seen: set = set()
    for img_id, img in images.items():
        if img_to_cat.get(img_id) != empty_cat_id:
            continue
        md_entry = get_md_entry(img)
        conf, _ = best_animal_det(md_entry)
        if conf < MD_EMPTY_CONF_MAX:
            seq_id = img.get(img_seq_key, img_id) if img_seq_key else img_id
            if seq_id not in seq_seen:
                seq_seen.add(seq_id)
                empty_imgs.append(
                    {"img": img, "conf": 0.0, "bbox": None, "seq_id": seq_id}
                )

    print(f"  {len(empty_imgs)} eligible empty (one per sequence)", end="")
    if len(empty_imgs) > EMPTY_CLASS_SAMPLES:
        empty_imgs = random.sample(empty_imgs, EMPTY_CLASS_SAMPLES)
        print(f"  → sampled {EMPTY_CLASS_SAMPLES}", end="")
    print()

# ---------------------------------------------------------------------------
# Stratified 70/15/15 split by sequence_id
# ---------------------------------------------------------------------------
by_class: dict[str, list] = defaultdict(list)
for label, item in sampled_animal:
    by_class[label].append(item)
if empty_imgs:
    by_class["empty"] = empty_imgs

manifest: list[dict] = []

for label, entries in sorted(by_class.items()):
    seq_ids = list({e["seq_id"] for e in entries})
    random.shuffle(seq_ids)
    n = len(seq_ids)
    n_train = max(1, round(n * TRAIN_RATIO))
    n_val = max(1, round(n * VAL_RATIO))
    seq_split: dict = {}
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
# Validation
# ---------------------------------------------------------------------------
print("\nValidation:")
seq_to_splits: dict = defaultdict(set)
for entry in manifest:
    seq_to_splits[entry["sequence_id"]].add(entry["split"])
leaked = {s: sp for s, sp in seq_to_splits.items() if len(sp) > 1}
print(
    f"  Sequence leakage: {'NONE' if not leaked else f'WARNING {len(leaked)} sequences span splits'}"
)

bad_conf = [
    e for e in manifest if e["label"] != "empty" and e["md_conf"] < MD_CONF_THRESHOLD
]
print(
    f"  Low-conf animal entries: {len(bad_conf)} {'OK' if not bad_conf else 'WARNING'}"
)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
split_label: dict = defaultdict(Counter)
for entry in manifest:
    split_label[entry["split"]][entry["label"]] += 1

all_labels = sorted({e["label"] for e in manifest})
print("\n" + "=" * 70)
print("SER SAMPLED MANIFEST SUMMARY")
print("=" * 70)
print(f"{'label':<25} {'train':>8} {'val':>8} {'test':>8} {'total':>8}")
print("-" * 58)
for label in all_labels:
    tr = split_label["train"][label]
    va = split_label["val"][label]
    te = split_label["test"][label]
    print(f"{label:<25} {tr:>8} {va:>8} {te:>8} {tr + va + te:>8}")
totals = {s: sum(c.values()) for s, c in split_label.items()}
grand = sum(totals.values())
print(
    f"{'TOTAL':<25} {totals.get('train', 0):>8} {totals.get('val', 0):>8} {totals.get('test', 0):>8} {grand:>8}"
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = SCRIPT_DIR / "ser_sampled_manifest.json"
with out_path.open("w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nSaved {len(manifest)} entries → {out_path}")
print("Done.")
