______________________________________________________________________

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
- 1K\<n\<10K

______________________________________________________________________

# Snapshot Safari SER (Serengeti) — Classroom Subset v1.0

## Summary

A curated subset of the Snapshot Safari 2024 Expansion SER (Serengeti National Park)
camera trap dataset, prepared for use in the CAS Deep Learning — Computer Vision
course exercises. Includes four dataset variants:

| Archive | Images | Description |
|------------------------------|-------:|------------------------------------------|
| `ser_balanced.tar.gz` | 1 850 | Balanced, ≤200/class, full frames |
| `ser_balanced_cropped.tar.gz`| 1 800 | Balanced, ≤200/class, MD-cropped |
| `ser_sampled.tar.gz` | 5 000 | Sampled, ≤1000/class, full frames |
| `ser_sampled_cropped.tar.gz` | 4 950 | Sampled, ≤1000/class, MD-cropped |

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

| Class | Train | Val | Test | Total |
|-----------------|------:|----:|-----:|------:|
| buffalo | 140 | 30 | 30 | 200 |
| elephant | 140 | 30 | 30 | 200 |
| empty | 35 | 8 | 7 | 50 |
| gazellegrants | 140 | 30 | 30 | 200 |
| gazellethomsons | 140 | 30 | 30 | 200 |
| hartebeest | 140 | 30 | 30 | 200 |
| impala | 140 | 30 | 30 | 200 |
| warthog | 140 | 30 | 30 | 200 |
| wildebeestblue | 140 | 30 | 30 | 200 |
| zebraplains | 140 | 30 | 30 | 200 |
| **Total** | 1295 | 278 | 277 | 1850 |

## ser_sampled Statistics

| Class | Train | Val | Test | Total |
|-----------------|------:|----:|-----:|------:|
| buffalo | 382 | 82 | 82 | 546 |
| elephant | 398 | 85 | 86 | 569 |
| empty | 35 | 8 | 7 | 50 |
| gazellegrants | 396 | 85 | 84 | 565 |
| gazellethomsons | 365 | 78 | 79 | 522 |
| hartebeest | 383 | 82 | 82 | 547 |
| impala | 379 | 81 | 82 | 542 |
| warthog | 387 | 83 | 83 | 553 |
| wildebeestblue | 382 | 82 | 81 | 545 |
| zebraplains | 393 | 84 | 84 | 561 |
| **Total** | 3500 | 750 | 750 | 5000 |

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
    "marco-willi/ser_balanced",
    "ser_balanced.tar.gz",
    repo_type="dataset",
)
with tarfile.open(archive) as tar:
    tar.extractall(DATA_PATH)
# → DATA_PATH/ser_balanced/{train,val,test}/<label>/*.jpg
```
