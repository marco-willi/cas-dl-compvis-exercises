"""Step 5 — Package the DeepFashion classroom subset as an internal-only ZIP.

Creates:
  data/deepfashion_classroom_v1_internal.zip   (data/deepfashion/ + metadata.csv + DATASET_CARD.md)
  data/deepfashion_classroom_v1_internal.zip.sha256

IMPORTANT: This ZIP is for internal course use only.
The DeepFashion dataset terms prohibit public redistribution.
Distribute only to enrolled course participants via a restricted Google Drive link.

Usage:
    python scripts/prepare_datasets/deepfashion/05_package_internal.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import textwrap
import zipfile
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = REPO_ROOT / "data" / "deepfashion"
METADATA_CSV = DATA_DIR / "metadata.csv"
MANIFEST_PATH = SCRIPT_DIR / "deepfashion_manifest.json"
ZIP_PATH = REPO_ROOT / "data" / "deepfashion_classroom_v1_internal.zip"
SHA256_PATH = ZIP_PATH.with_suffix(".zip.sha256")
DATASET_CARD_PATH = DATA_DIR / "DATASET_CARD.md"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
missing = []
if not DATA_DIR.exists():
    missing.append(str(DATA_DIR))
if not METADATA_CSV.exists():
    missing.append(str(METADATA_CSV))
if missing:
    print("ERROR: Missing required files/directories:")
    for m in missing:
        print(f"  {m}")
    print("Run 03_copy_images.py first.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Read metadata for dataset card stats
# ---------------------------------------------------------------------------
print("Reading metadata...")
with METADATA_CSV.open(newline="") as f:
    rows = list(csv.DictReader(f))

total_images = len(rows)
split_counts: Counter = Counter(r["split"] for r in rows)
label_counts: Counter = Counter(r["label"] for r in rows)
splits = ["train", "val", "test"]

print(f"  Total images: {total_images}")
for split in splits:
    print(f"  {split}: {split_counts.get(split, 0)}")

# Item-level counts
item_split: dict[str, str] = {}
for r in rows:
    item_split[r["item_id"]] = r["split"]
item_counts_by_split: Counter = Counter(item_split.values())
total_items = len(item_split)

# Load manifest for avg images/item
if MANIFEST_PATH.exists():
    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)
    item_image_counts: Counter = Counter(e["item_id"] for e in manifest)
    avg_images_per_item = (
        sum(item_image_counts.values()) / len(item_image_counts)
        if item_image_counts
        else 0.0
    )
else:
    avg_images_per_item = 0.0

# ---------------------------------------------------------------------------
# Write DATASET_CARD.md (inside the ZIP)
# ---------------------------------------------------------------------------
label_table_rows = "\n".join(
    f"| {label} | {label_counts[label]} |" for label in sorted(label_counts)
)
split_table_rows = "\n".join(
    f"| {s} | {item_counts_by_split.get(s, 0)} | {split_counts.get(s, 0)} |"
    for s in splits
)

dataset_card = textwrap.dedent(f"""\
    # DeepFashion Classroom Subset — Dataset Card

    > **INTERNAL USE ONLY — DO NOT REDISTRIBUTE**
    >
    > This dataset is derived from the DeepFashion In-Shop Clothes Retrieval Benchmark.
    > The original dataset terms (MMLAB/CUHK) prohibit copying, publishing, or
    > distributing images or derived data beyond the authorised research group.
    > This subset is licensed for use by enrolled participants of the CAS Deep Learning
    > — Computer Vision course only. Do not share, publish, or upload to any public host.

    ---

    ## Source

    - **Benchmark:** DeepFashion In-Shop Clothes Retrieval Benchmark
    - **Provided by:** MMLAB, The Chinese University of Hong Kong (CUHK)
    - **Citation:** Liu, Z. et al. "DeepFashion: Powering Robust Clothes Recognition and
      Retrieval with Rich Annotations." CVPR 2016.
    - **Original dataset URL:** https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html

    ## Coarse Taxonomy

    The original fine-grained clothing categories are mapped to {len(label_counts)} coarse classes:

    | Coarse label | Images |
    |---|---|
    {label_table_rows}

    ## Split Sizes

    Items are split by item ID so that all images of one item stay in the same split.
    This prevents retrieval leakage (a query image and its ground-truth matches are
    never in different splits).

    | Split | Items | Images |
    |---|---|---|
    {split_table_rows}
    | **Total** | **{total_items}** | **{total_images}** |

    ## Image Details

    - **Resolution:** original resolution preserved (no pre-resize at curation time)
    - **Format:** JPEG, RGB
    - **Average images per item:** {avg_images_per_item:.1f}

    ## Directory Layout

    ```
    deepfashion/
    ├── train/
    │   ├── <label>/
    │   │   └── <item_id>_<frame>.jpg
    ├── val/
    │   └── <label>/...
    ├── test/
    │   └── <label>/...
    └── metadata.csv      # image_id, item_id, file_name, split, label, width, height
    ```

    `torchvision.datasets.ImageFolder` loads each split directly.

    ## Redistribution Restrictions

    **This dataset MUST NOT be:**
    - Uploaded to Hugging Face Hub, Kaggle, or any public data repository
    - Shared via public links, public cloud storage, or social media
    - Used for any commercial purpose

    Distribute only to enrolled course participants via a restricted (non-public)
    Google Drive link provided by the course instructor.
""")

DATASET_CARD_PATH.write_text(dataset_card)
print(f"\nWrote DATASET_CARD.md: {DATASET_CARD_PATH}")

# ---------------------------------------------------------------------------
# Create ZIP
# ---------------------------------------------------------------------------
print(f"\nCreating ZIP: {ZIP_PATH}")
files_added = 0

with zipfile.ZipFile(
    ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
) as zf:
    # Add all images and metadata.csv under deepfashion/
    for file_path in sorted(DATA_DIR.rglob("*")):
        if not file_path.is_file():
            continue
        arc_name = "deepfashion/" + file_path.relative_to(DATA_DIR).as_posix()
        zf.write(file_path, arc_name)
        files_added += 1
        if files_added % 500 == 0:
            print(f"  Added {files_added} files...")

print(f"  Total files added: {files_added}")

zip_size_mb = ZIP_PATH.stat().st_size / 1024 / 1024
print(f"  ZIP size: {zip_size_mb:.1f} MB")

# ---------------------------------------------------------------------------
# SHA-256 checksum
# ---------------------------------------------------------------------------
print("\nComputing SHA-256 checksum...")
sha256 = hashlib.sha256()
with ZIP_PATH.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        sha256.update(chunk)
digest = sha256.hexdigest()

SHA256_PATH.write_text(f"{digest}  {ZIP_PATH.name}\n")
print(f"  SHA-256: {digest}")
print(f"  Written to: {SHA256_PATH}")

# ---------------------------------------------------------------------------
# Smoke-test: verify ZIP extracts cleanly
# ---------------------------------------------------------------------------
print("\nVerifying ZIP integrity...")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    bad = zf.testzip()
if bad:
    print(f"  WARNING: first bad file in ZIP: {bad}")
else:
    print("  ZIP integrity OK")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PACKAGING SUMMARY")
print("=" * 60)
print(f"  ZIP:       {ZIP_PATH}")
print(f"  Size:      {zip_size_mb:.1f} MB")
print(f"  Files:     {files_added}")
print(f"  SHA-256:   {digest}")
print(
    "\nNext steps:"
    "\n  1. Upload deepfashion_classroom_v1_internal.zip to a RESTRICTED Google Drive folder."
    "\n  2. Set sharing to 'Anyone with the link' only if you have verified the"
    "\n     participants list, or use 'Specific people' for maximum control."
    "\n  3. Paste the Drive file ID into the load_deepfashion() helper in the exercise notebooks."
    "\n  4. Test gdown download in a fresh Colab before distributing to students."
)
