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

| Class | Train | Val | Test | Total |
|--------------------|------:|----:|-----:|------:|
| empty | 1870 | 401 | 401 | 2672 |
| gemsbokoryx | 348 | 75 | 74 | 497 |
| birdother | 76 | 16 | 16 | 108 |
| steenbok | 41 | 9 | 8 | 58 |
| ostrich | 32 | 7 | 7 | 46 |
| jackalblackbacked | 27 | 6 | 6 | 39 |
| **Total** | 2394 | 514 | 512 | 3420 |

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
