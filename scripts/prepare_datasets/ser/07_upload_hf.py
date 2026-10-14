"""Step 7 — Package and upload SER dataset to Hugging Face Hub.

Creates four archives under data/:
    ser_balanced.tar.gz           (full-frame balanced images)
    ser_balanced_cropped.tar.gz   (MD-cropped balanced images)
    ser_sampled.tar.gz            (full-frame sampled images, realistic distribution)
    ser_sampled_cropped.tar.gz    (MD-cropped sampled images)

Each archive includes its metadata CSV and README.

Uploads all four archives + dataset card as README.md to:
    https://huggingface.co/datasets/marco-willi/camera-trap-ser

Tags repo at v1.0 and verifies unauthenticated download.

Requires:
    huggingface-cli login   (or HF_TOKEN env var)

Usage:
    python scripts/prepare_datasets/ser/07_upload_hf.py
    python scripts/prepare_datasets/ser/07_upload_hf.py --package-only
    python scripts/prepare_datasets/ser/07_upload_hf.py --upload-only
"""

import argparse
import hashlib
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
SER_ROOT = DATA_DIR / "ser"
SCRIPT_DIR = Path(__file__).resolve().parent
CARD_PATH = SCRIPT_DIR / "dataset_card.md"
REPO_ID = "marco-willi/camera-trap-ser"

ARCHIVES = {
    "ser_balanced": SER_ROOT / "ser_balanced",
    "ser_balanced_cropped": SER_ROOT / "ser_balanced_cropped",
    "ser_sampled": SER_ROOT / "ser_sampled",
    "ser_sampled_cropped": SER_ROOT / "ser_sampled_cropped",
}

DATASET_CARD = """\
---
license: cdla-permissive-1.0
task_categories:
  - image-classification
tags:
  - camera-trap
  - wildlife
  - serengeti
  - snapshot-safari
  - megadetector
pretty_name: Snapshot Safari SER — Classroom Subset v1.0
size_categories:
  - 1K<n<10K
---

# Snapshot Safari SER (Serengeti) — Classroom Subset v1.0

## Summary

A curated subset of the Snapshot Safari 2024 Expansion SER (Serengeti National Park)
camera trap dataset, prepared for use in the CAS Deep Learning — Computer Vision
course exercises. Includes four dataset variants:

| Archive                      | Images | Description                              |
|------------------------------|-------:|------------------------------------------|
| `ser_balanced.tar.gz`        |  1 850 | Balanced, ≤200/class, full frames        |
| `ser_balanced_cropped.tar.gz`|  1 800 | Balanced, ≤200/class, MD-cropped         |
| `ser_sampled.tar.gz`         |  5 000 | Sampled, ≤1000/class, full frames        |
| `ser_sampled_cropped.tar.gz` |  4 950 | Sampled, ≤1000/class, MD-cropped         |

## Source

- **Dataset:** Snapshot Safari 2024 Expansion — SER (Serengeti) subset
- **URL:** https://lila.science/datasets/snapshot-safari-2024-expansion/
- **License:** Community Data License Agreement — Permissive variant 1.0
- **Attribution:** Snapshot Safari / University of Minnesota Lion Center

## MegaDetector

Pre-computed MegaDetector v1000-redwood RDE-filtered results from LILA Science:
`snapshot-safari-2024-expansion-SER-subset-v1000.0.0-redwood_detections.threshold.filtered.json.zip`

Used to filter frames (conf ≥ 0.8), select the best frame per sequence, and
crop images to the primary detected animal bounding box (10% padding).

## Species

buffalo, elephant, empty, gazellegrants, gazellethomsons, hartebeest, impala,
warthog, wildebeestblue, zebraplains

## ser_balanced Statistics

| Class           | Train | Val | Test | Total |
|-----------------|------:|----:|-----:|------:|
| buffalo         |   140 |  30 |   30 |   200 |
| elephant        |   140 |  30 |   30 |   200 |
| empty           |    35 |   8 |    7 |    50 |
| gazellegrants   |   140 |  30 |   30 |   200 |
| gazellethomsons |   140 |  30 |   30 |   200 |
| hartebeest      |   140 |  30 |   30 |   200 |
| impala          |   140 |  30 |   30 |   200 |
| warthog         |   140 |  30 |   30 |   200 |
| wildebeestblue  |   140 |  30 |   30 |   200 |
| zebraplains     |   140 |  30 |   30 |   200 |
| **Total**       |  1295 | 278 |  277 |  1850 |

## ser_sampled Statistics

| Class           | Train | Val | Test | Total |
|-----------------|------:|----:|-----:|------:|
| buffalo         |   382 |  82 |   82 |   546 |
| elephant        |   398 |  85 |   86 |   569 |
| empty           |    35 |   8 |    7 |    50 |
| gazellegrants   |   396 |  85 |   84 |   565 |
| gazellethomsons |   365 |  78 |   79 |   522 |
| hartebeest      |   383 |  82 |   82 |   547 |
| impala          |   379 |  81 |   82 |   542 |
| warthog         |   387 |  83 |   83 |   553 |
| wildebeestblue  |   382 |  82 |   81 |   545 |
| zebraplains     |   393 |  84 |   84 |   561 |
| **Total**       |  3500 | 750 |  750 |  5000 |

**Note:** ~56% of images are IR/night (near-infrared, nearly greyscale).

## Curation Details

- **Deduplication:** one image per sequence (highest MD animal confidence frame)
- **Animal filter:** MD animal confidence ≥ 0.8
- **Empty filter:** max MD animal confidence < 0.2
- **Split strategy:** stratified 70/15/15 by sequence ID — no sequence spans splits
- **Image resolution:** resized to max 1024 px on longer side, JPEG quality 92
- **Format:** ImageFolder layout — `<split>/<label>/<filename>.jpg`

## Usage

```python
from huggingface_hub import hf_hub_download
import tarfile

archive = hf_hub_download(
    "marco-willi/camera-trap-ser",
    "ser_balanced.tar.gz",
    repo_type="dataset",
)
with tarfile.open(archive) as tar:
    tar.extractall(DATA_PATH)
# → DATA_PATH/ser_balanced/{train,val,test}/<label>/*.jpg
```
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_archive(src_dir: Path, archive_path: Path):
    print(f"  Creating {archive_path.name} from {src_dir.name}/ ...")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(src_dir, arcname=src_dir.name)
    size_mb = archive_path.stat().st_size / 1e6
    print(f"  Size: {size_mb:.1f} MB")


def write_checksum(archive_path: Path) -> str:
    digest = sha256(archive_path)
    checksum_path = archive_path.with_suffix(".gz.sha256")
    checksum_path.write_text(f"{digest}  {archive_path.name}\n")
    print(f"  SHA-256: {digest[:16]}...")
    return digest


# ---------------------------------------------------------------------------
# Package
# ---------------------------------------------------------------------------
def package():
    print("\n=== Packaging ===")
    CARD_PATH.write_text(DATASET_CARD)
    print(f"  Written: {CARD_PATH.name}")

    for name, src_dir in ARCHIVES.items():
        if not src_dir.exists():
            print(f"  SKIP (not found): {src_dir}")
            continue
        # Copy card into sub-dataset dir
        (src_dir / "README.md").write_text(DATASET_CARD)
        archive_path = DATA_DIR / f"{name}.tar.gz"
        make_archive(src_dir, archive_path)
        write_checksum(archive_path)

    print("\nPackaging complete.")
    for name in ARCHIVES:
        p = DATA_DIR / f"{name}.tar.gz"
        if p.exists():
            print(f"  {name}.tar.gz: {p.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def upload():
    from huggingface_hub import HfApi, create_repo, hf_hub_download

    api = HfApi()
    print(f"\n=== Uploading to {REPO_ID} ===")
    create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)

    for name in ARCHIVES:
        archive = DATA_DIR / f"{name}.tar.gz"
        if not archive.exists():
            print(f"  SKIP (not found): {archive.name}")
            continue
        size_mb = archive.stat().st_size / 1e6
        print(f"  Uploading {archive.name} ({size_mb:.0f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(archive),
            path_in_repo=archive.name,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        checksum_path = archive.with_suffix(".gz.sha256")
        if checksum_path.exists():
            api.upload_file(
                path_or_fileobj=str(checksum_path),
                path_in_repo=checksum_path.name,
                repo_id=REPO_ID,
                repo_type="dataset",
            )

    print("  Uploading README.md (dataset card) ...")
    api.upload_file(
        path_or_fileobj=str(CARD_PATH),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    print("  Tagging at v1.0 ...")
    try:
        api.create_tag(REPO_ID, tag="v1.0", repo_type="dataset")
    except Exception as e:
        print(f"  Tag: {e}")

    print(f"\n  Uploaded: https://huggingface.co/datasets/{REPO_ID}")

    print("\n  Verifying unauthenticated download ...")
    path = hf_hub_download(REPO_ID, "ser_balanced.tar.gz", repo_type="dataset")
    print(f"  Downloaded: {path}")
    print("  OK.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--upload-only", action="store_true")
    args = parser.parse_args()

    if args.upload_only:
        upload()
    elif args.package_only:
        package()
    else:
        package()
        upload()
