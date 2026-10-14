"""Step 1 — Download Snapshot Safari SER metadata and MegaDetector results.

Downloads the full Snapshot Safari 2024 Expansion metadata ZIP from GCS,
extracts only the SER (Serengeti) entries, and saves a smaller
ser_metadata.json to .cache/. Also downloads the MegaDetector RDE-filtered
results for the SER subset.

Writes:
    .cache/ser_metadata.json   — SER-only COCO Camera Traps JSON
    .cache/ser_md_results.json — MegaDetector detections for SER images

Usage:
    python scripts/prepare_datasets/ser/01_download_metadata.py
"""

import json
import zipfile
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METADATA_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "snapshot-safari-2024-expansion/snapshot_safari_2024_metadata.zip"
)
MD_URL = (
    "https://lila.science/public/lila-md-results/"
    "snapshot-safari-2024-expansion-SER-subset-v1000.0.0-redwood_detections"
    ".threshold.filtered.json.zip"
)
SER_PREFIX = "SER/"

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

METADATA_ZIP_CACHE = CACHE_DIR / "snapshot_safari_2024_metadata.zip"
MD_ZIP_CACHE = CACHE_DIR / "ser_md_results.zip"
SER_METADATA_OUT = CACHE_DIR / "ser_metadata.json"
MD_OUT = CACHE_DIR / "ser_md_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_with_cache(url: str, dest: Path, label: str) -> None:
    """Stream-download url to dest, skip if already present."""
    if dest.exists():
        print(f"  [cached] {label}: {dest}")
        return
    print(f"  Downloading {label}...")
    print(f"    URL: {url}")
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(
                        f"    {downloaded >> 20} / {total >> 20} MB ({pct:.0f}%)",
                        end="\r",
                    )
    print(f"    Done: {dest} ({dest.stat().st_size >> 20} MB)          ")


# ---------------------------------------------------------------------------
# Step 1a — Download and extract main metadata (filter to SER)
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1a — Snapshot Safari 2024 Expansion metadata")
print("=" * 60)
download_with_cache(METADATA_URL, METADATA_ZIP_CACHE, "metadata ZIP")

print(f"\nExtracting and filtering to SER images (prefix '{SER_PREFIX}')...")
with zipfile.ZipFile(METADATA_ZIP_CACHE) as zf:
    json_names = [n for n in zf.namelist() if n.endswith(".json")]
    print(f"  JSON files in ZIP: {json_names}")
    assert json_names, "No JSON file found in metadata ZIP."
    json_name = json_names[0]
    print(f"  Reading: {json_name}")
    raw = json.loads(zf.read(json_name))

print(f"\nFull dataset top-level keys: {list(raw.keys())}")

all_images = raw.get("images", [])
all_annotations = raw.get("annotations", [])
all_categories = raw.get("categories", [])
sequences = raw.get("sequences", [])

print(f"  Total images:      {len(all_images):>10,}")
print(f"  Total annotations: {len(all_annotations):>10,}")
print(f"  Total categories:  {len(all_categories):>10,}")
print(f"  Total sequences:   {len(sequences):>10,}")

# Detect SER images by file_name prefix
ser_images = [
    img for img in all_images if img.get("file_name", "").startswith(SER_PREFIX)
]
print(f"\n  SER images (prefix '{SER_PREFIX}'): {len(ser_images):,}")

if not ser_images:
    # Try alternative: check 'location' field or other indicators
    print("  WARNING: No images with SER/ prefix found. Checking sample filenames:")
    for img in all_images[:5]:
        print(f"    {img.get('file_name', 'NO file_name')}")
    raise SystemExit(
        "Could not identify SER images. Inspect the JSON structure and update SER_PREFIX."
    )

ser_image_ids = {img["id"] for img in ser_images}

# Filter annotations to SER images (handle image_id or seq_id based annotations)
sample_ann = all_annotations[0] if all_annotations else {}
print(f"\n  Sample annotation keys: {list(sample_ann.keys())}")

has_image_id = "image_id" in sample_ann
has_seq_id = "seq_id" in sample_ann or "sequence_id" in sample_ann
seq_key_ann = (
    "seq_id"
    if "seq_id" in sample_ann
    else "sequence_id"
    if "sequence_id" in sample_ann
    else None
)

print(f"  Annotations have image_id: {has_image_id}")
print(f"  Annotations have seq_id:   {seq_key_ann}")

# Check sequence key on images
sample_img = ser_images[0]
print(f"\n  Sample SER image keys: {list(sample_img.keys())}")
img_seq_key = (
    "seq_id"
    if "seq_id" in sample_img
    else "sequence_id"
    if "sequence_id" in sample_img
    else None
)
print(f"  Sequence key on images: {img_seq_key}")

# Filter annotations to SER
if has_image_id:
    ser_annotations = [
        ann for ann in all_annotations if ann["image_id"] in ser_image_ids
    ]
else:
    # Sequence-level: collect SER sequence IDs, then filter
    ser_seq_ids = {img.get(img_seq_key) for img in ser_images if img_seq_key}
    ser_annotations = [
        ann for ann in all_annotations if ann.get(seq_key_ann) in ser_seq_ids
    ]

print(f"\n  SER annotations: {len(ser_annotations):,}")

# Filter sequences to SER (if sequences table present)
if sequences and img_seq_key:
    ser_seq_ids_set = {img.get(img_seq_key) for img in ser_images}
    ser_sequences = [s for s in sequences if s.get("id") in ser_seq_ids_set]
    print(f"  SER sequences:   {len(ser_sequences):,}")
else:
    ser_sequences = []

# Build and save filtered SER metadata
ser_metadata = {
    "info": raw.get("info", {}),
    "categories": all_categories,
    "images": ser_images,
    "annotations": ser_annotations,
}
if ser_sequences:
    ser_metadata["sequences"] = ser_sequences

with SER_METADATA_OUT.open("w") as f:
    json.dump(ser_metadata, f)
print(f"\n  Saved SER metadata to: {SER_METADATA_OUT}")
print(f"  File size: {SER_METADATA_OUT.stat().st_size >> 20} MB")

# ---------------------------------------------------------------------------
# Step 1b — Download MegaDetector RDE results
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 1b — MegaDetector RDE results for SER subset")
print("=" * 60)
download_with_cache(MD_URL, MD_ZIP_CACHE, "MD results ZIP")

print("\nExtracting MD results...")
with zipfile.ZipFile(MD_ZIP_CACHE) as zf:
    json_names = [n for n in zf.namelist() if n.endswith(".json")]
    print(f"  JSON files in ZIP: {json_names}")
    assert json_names, "No JSON file found in MD results ZIP."
    md_json_name = json_names[0]
    print(f"  Reading: {md_json_name}")
    md_data = json.loads(zf.read(md_json_name))

with MD_OUT.open("w") as f:
    json.dump(md_data, f)
print(f"  Saved MD results to: {MD_OUT}")
print(f"  File size: {MD_OUT.stat().st_size >> 20} MB")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# MD stats
md_images = md_data.get("images", [])
print(f"  MD result images:      {len(md_images):,}")

# Count detections
total_detections = sum(len(img.get("detections", [])) for img in md_images)
animal_detections = sum(
    1
    for img in md_images
    for det in img.get("detections", [])
    if str(det.get("category", "")) == "1" and det.get("conf", 0) >= 0.8
)
print(f"  Total MD detections:   {total_detections:,}")
print(f"  Animal det (conf≥0.8): {animal_detections:,}")
print(
    f"  Images w/ animal det:  {sum(1 for img in md_images if any(str(d.get('category', '')) == '1' and d.get('conf', 0) >= 0.8 for d in img.get('detections', []))):,}"
)

# SER metadata stats
categories = {c["id"]: c["name"] for c in all_categories}
print(f"\n  SER images:            {len(ser_images):,}")
print(f"  SER annotations:       {len(ser_annotations):,}")
print(f"  Categories total:      {len(categories):,}")

print("\nDone.")
