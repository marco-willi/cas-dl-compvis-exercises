"""Step C2 — Package datasets as tar.gz and write dataset cards.

For each dataset:
1. Write a dataset_card.md
2. Copy the card into the data directory as README.md
3. Create a tar.gz archive (top-level dir = the dataset's data folder name)
4. Compute SHA-256 checksum

Usage:
    python scripts/prepare_datasets/C2_package.py kgalagadi
    python scripts/prepare_datasets/C2_package.py cct20
    python scripts/prepare_datasets/C2_package.py abo_furniture
    python scripts/prepare_datasets/C2_package.py cats_vs_dogs
    python scripts/prepare_datasets/C2_package.py concrete_cracks
    python scripts/prepare_datasets/C2_package.py eurosat
    python scripts/prepare_datasets/C2_package.py osu_small_animals
    python scripts/prepare_datasets/C2_package.py all
"""

import hashlib
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Dataset cards
# ---------------------------------------------------------------------------
CARDS = {
    "kgalagadi": """\
# Snapshot Kgalagadi — Classroom Subset v1.0

## Summary

A curated subset of the Snapshot Kgalagadi S1 camera trap dataset for use in
the CAS Deep Learning — Computer Vision course exercises.

## Source

- **Dataset:** Snapshot Kgalagadi Season 1 (Snapshot Safari series)
- **URL:** https://lila.science/datasets/snapshot-kgalagadi
- **License:** Community Data License Agreement — Permissive variant
- **Attribution:** University of Minnesota Lion Center / Snapshot Safari

## Curation

- **Selected classes:** empty, gemsbokoryx, birdother, steenbok, ostrich, jackalblackbacked
- **Deduplication:** one image per sequence (first frame, lowest frame_num)
- **Human images:** excluded
- **Split strategy:** stratified 70/15/15 by sequence ID — no sequence spans splits
- **Image resolution:** original (resized to max 1024 px on the longest side, aspect ratio preserved)
- **Format:** JPEG, ImageFolder layout (`<split>/<label>/<filename>.jpg`)

## Statistics

| Class              | Train | Val | Test | Total |
|--------------------|------:|----:|-----:|------:|
| empty              | 1870  | 401 | 401  | 2672  |
| gemsbokoryx        |  348  |  75 |  74  |  497  |
| birdother          |   76  |  16 |  16  |  108  |
| steenbok           |   41  |   9 |   8  |   58  |
| ostrich            |   32  |   7 |   7  |   46  |
| jackalblackbacked  |   27  |   6 |   6  |   39  |
| **Total**          | 2394  | 514 | 512  | 3420  |

**Note:** The empty class dominates (~78%). This is intentional — it reflects
real-world camera trap data and is used to teach class imbalance in Exercise 1.

## Sidecar files

- `metadata.csv` — per-image record with `image_id`, `file_name`, `split`, `label`, `width`, `height`

## Usage in notebooks

```python
from huggingface_hub import hf_hub_download
import tarfile

archive = hf_hub_download(
    "marco-willi/camera-trap-kgalagadi",
    "kgalagadi.tar.gz",
    repo_type="dataset",
)
with tarfile.open(archive) as tar:
    tar.extractall(DATA_PATH)
```
""",
    "cct20": """\
# Caltech Camera Traps CCT20 — Classroom Subset v1.0

## Summary

A curated, balanced subset of the Caltech Camera Traps CCT20 benchmark for use
in the CAS Deep Learning — Computer Vision course exercises.

## Source

- **Dataset:** Caltech Camera Traps — CCT20 benchmark subset
- **URL:** https://lila.science/datasets/caltech-camera-traps
- **License:** Community Data License Agreement — Permissive variant 1.0
- **Attribution:** Beery et al., "Recognition in Terra Incognita", ECCV 2018

## Curation

- **Selected classes:** bobcat, cat, coyote, empty, opossum, rabbit, raccoon, squirrel
- **Sampling:** 200 images per class (random seed 42), drawn from CCT20 images
- **Split strategy:** stratified 80/10/10 per class
- **Image resolution:** original CCT20 resolution (pre-downsampled by LILA to ≤1024 px)
- **Format:** JPEG, ImageFolder layout (`<split>/<label>/<filename>.jpg`)

## Statistics

| Class    | Train | Val | Test | Total |
|----------|------:|----:|-----:|------:|
| bobcat   |   160 |  20 |   20 |   200 |
| cat      |   160 |  20 |   20 |   200 |
| coyote   |   160 |  20 |   20 |   200 |
| empty    |   160 |  20 |   20 |   200 |
| opossum  |   160 |  20 |   20 |   200 |
| rabbit   |   160 |  20 |   20 |   200 |
| raccoon  |   160 |  20 |   20 |   200 |
| squirrel |   160 |  20 |   20 |   200 |
| **Total**| 1280  | 160 |  160 |  1600 |

## Sidecar files

- `metadata.csv` — per-image record with `image_id`, `file_name`, `split`, `label`, `width`, `height`

## Usage in notebooks

```python
from huggingface_hub import hf_hub_download
import tarfile

archive = hf_hub_download(
    "marco-willi/camera-trap-cct20",
    "cct20.tar.gz",
    repo_type="dataset",
)
with tarfile.open(archive) as tar:
    tar.extractall(DATA_PATH)
```
""",
}


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Dataset configurations
#
# Most datasets have data_dir == archive_name == arcname. ABO is the exception:
# its data lives at data/abo/ but the public archive is abo_furniture.tar.gz —
# we still want the tar's top-level directory to be `abo/` so the notebook can
# extract with extracted_subdir="abo".
# ---------------------------------------------------------------------------
DATASETS = {
    "kgalagadi": {"data_subdir": "kgalagadi", "arcname": "kgalagadi"},
    "cct20": {"data_subdir": "cct20", "arcname": "cct20"},
    "abo_furniture": {"data_subdir": "abo", "arcname": "abo"},
    # Optional alternative datasets (GDrive-only, no HF mirror due to licensing)
    "cats_vs_dogs": {"data_subdir": "cats_vs_dogs", "arcname": "cats_vs_dogs"},
    "concrete_cracks": {"data_subdir": "concrete_cracks", "arcname": "concrete_cracks"},
    "eurosat": {"data_subdir": "eurosat", "arcname": "eurosat"},
    "osu_small_animals": {
        "data_subdir": "osu_small_animals",
        "arcname": "osu_small_animals",
    },
}


def _card_text(dataset: str) -> str:
    """Return the dataset card text, preferring an inline CARDS entry,
    otherwise reading from <dataset>_card.md alongside this script."""
    if dataset in CARDS:
        return CARDS[dataset]
    card_path = SCRIPT_DIR / f"{dataset}_card.md"
    assert card_path.exists(), f"No CARDS entry and no file at {card_path}"
    return card_path.read_text()


def package(dataset: str):
    cfg = DATASETS[dataset]
    data_dir = DATA_DIR / cfg["data_subdir"]
    assert data_dir.exists(), f"Dataset directory not found: {data_dir}"

    # Write README.md into the data directory
    card = _card_text(dataset)
    readme_path = data_dir / "README.md"
    readme_path.write_text(card)
    print(f"  Written: {readme_path}")

    # Write card to scripts dir for reference (only if it came from CARDS)
    if dataset in CARDS:
        card_path = SCRIPT_DIR / f"{dataset}_card.md"
        card_path.write_text(card)
        print(f"  Written: {card_path}")

    # Create tar.gz with arcname controlling the top-level directory inside it
    archive_path = DATA_DIR / f"{dataset}.tar.gz"
    print(f"  Creating archive: {archive_path} (top-level: {cfg['arcname']}/) ...")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(data_dir, arcname=cfg["arcname"])
    size_mb = archive_path.stat().st_size / 1e6
    print(f"  Archive size: {size_mb:.1f} MB")

    # Checksum
    digest = sha256(archive_path)
    checksum_path = DATA_DIR / f"{dataset}.tar.gz.sha256"
    checksum_path.write_text(f"{digest}  {dataset}.tar.gz\n")
    print(f"  SHA-256: {digest}")
    print(f"  Written: {checksum_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    targets = sys.argv[1:] if sys.argv[1:] else ["all"]
    if targets == ["all"]:
        targets = list(DATASETS.keys())

    for ds in targets:
        assert ds in DATASETS, (
            f"Unknown dataset '{ds}'. Choose from: {list(DATASETS.keys())}"
        )
        print(f"\n{'=' * 60}")
        print(f"Packaging: {ds}")
        print(f"{'=' * 60}")
        package(ds)

    print("\nDone.")
