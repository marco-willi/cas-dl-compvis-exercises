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
  pretty_name: Snapshot Safari SER Sampled — Classroom Subset v1.0
  size_categories:
- 1K\<n\<10K

______________________________________________________________________

# Snapshot Safari SER (Serengeti) — Sampled Classroom Subset v1.0

## Summary

A curated, realistically distributed subset of the Snapshot Safari 2024 Expansion SER
(Serengeti National Park) camera trap dataset, prepared for use in the CAS Deep Learning —
Computer Vision course exercises. Unlike `ser_balanced`, the class distribution reflects
real-world Serengeti encounter rates: wildebeest and zebra dominate, while rarer species
(impala, warthog, gazelle grants) appear infrequently.

| Archive | Images | Description |
|---|---:|---|
| `ser_sampled.tar.gz` | 4 999 | Realistic distribution, full frames |
| `ser_sampled_cropped.tar.gz` | 4 949 | Realistic distribution, MD-cropped |

## Source

- **Dataset:** Snapshot Safari 2024 Expansion — SER (Serengeti) subset
- **URL:** https://lila.science/datasets/snapshot-safari-2024-expansion/
- **License:** Community Data License Agreement — Permissive variant 1.0
- **Attribution:** Snapshot Safari / University of Minnesota Lion Center

## MegaDetector

Pre-computed MegaDetector v1000-redwood RDE-filtered results from LILA Science:
`snapshot-safari-2024-expansion-SER-subset-v1000.0.0-redwood_detections.threshold.filtered.json.zip`

Used to filter frames (conf ≥ 0.8) and select the best frame per sequence.
The `_cropped` variant additionally crops each image to the primary detected animal
bounding box (10% padding).

## Species

buffalo, elephant, empty, gazellegrants, gazellethomsons, hartebeest, impala,
warthog, wildebeestblue, zebraplains

## Statistics — `ser_sampled`

| Class | Train | Val | Test | Total |
|-----------------|------:|----:|-----:|------:|
| wildebeestblue | 1 244 | 267 | 266 | 1 777 |
| zebraplains | 919 | 197 | 197 | 1 313 |
| gazellethomsons | 724 | 155 | 156 | 1 035 |
| buffalo | 146 | 31 | 31 | 208 |
| elephant | 115 | 25 | 25 | 165 |
| hartebeest | 115 | 25 | 24 | 164 |
| gazellegrants | 74 | 16 | 16 | 106 |
| warthog | 66 | 14 | 15 | 95 |
| impala | 60 | 13 | 13 | 86 |
| empty | 35 | 8 | 7 | 50 |
| **Total** | **3 498** | **751** | **750** | **4 999** |

## Statistics — `ser_sampled_cropped`

Same splits and class labels as `ser_sampled`. The `empty` class is excluded from
the cropped variant (no animal detection box available). Total: 4 949 images.

**Note:** ~56% of images are IR/night (near-infrared, nearly greyscale).

## Curation Details

- **Deduplication:** one image per sequence (highest MD animal confidence frame)
- **Animal filter:** MD animal confidence ≥ 0.8
- **Empty filter:** max MD animal confidence < 0.2
- **Sampling:** proportional to real Serengeti encounter rates (no per-class cap)
- **Split strategy:** stratified 70/15/15 by sequence ID — no sequence spans splits
- **Image resolution:** resized to max 1024 px on longer side, JPEG quality 92
- **Format:** ImageFolder layout — `<split>/<label>/<filename>.jpg`

## Usage

```python
from huggingface_hub import hf_hub_download
import tarfile

# Full-frame version
archive = hf_hub_download(
    "marco-willi/ser_sampled",
    "ser_sampled.tar.gz",
    repo_type="dataset",
)
with tarfile.open(archive) as tar:
    tar.extractall(DATA_PATH)
# → DATA_PATH/ser_sampled/{train,val,test}/<label>/*.jpg

# MD-cropped version (recommended for Colab — smaller download)
archive = hf_hub_download(
    "marco-willi/ser_sampled_cropped",
    "ser_sampled_cropped.tar.gz",
    repo_type="dataset",
)
with tarfile.open(archive) as tar:
    tar.extractall(DATA_PATH)
# → DATA_PATH/ser_sampled_cropped/{train,val,test}/<label>/*.jpg
```
