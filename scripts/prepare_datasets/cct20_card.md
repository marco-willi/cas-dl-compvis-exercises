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

| Class | Train | Val | Test | Total |
|----------|------:|----:|-----:|------:|
| bobcat | 160 | 20 | 20 | 200 |
| cat | 160 | 20 | 20 | 200 |
| coyote | 160 | 20 | 20 | 200 |
| empty | 160 | 20 | 20 | 200 |
| opossum | 160 | 20 | 20 | 200 |
| rabbit | 160 | 20 | 20 | 200 |
| raccoon | 160 | 20 | 20 | 200 |
| squirrel | 160 | 20 | 20 | 200 |
| **Total**| 1280 | 160 | 160 | 1600 |

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
