# ABO Furniture — Classroom Subset v1.0

## Summary

A curated subset of the Amazon Berkeley Objects (ABO) dataset for use in the
CAS Deep Learning — Computer Vision course exercises. Images show e-commerce
product photographs of furniture across 6 categories, with multiple views per
product item — making it well-suited for both classification and image retrieval
exercises.

## Source

- **Dataset:** Amazon Berkeley Objects (ABO)
- **URL:** https://amazon-berkeley-objects.s3.amazonaws.com/index.html
- **License:** Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)
- **Attribution:** Collins et al., "ABO: Dataset and Benchmarks for Real-World
  3D Object Understanding", CVPR 2022

## Curation

- **Selected classes:** bed, chair, lamp, sofa, storage, table
- **Mapping:** product types mapped to 6 canonical labels via exact product_type matching
- **Views per item:** primary image + up to 3 additional views (view_index 0–3)
- **Deduplication:** global image_id deduplication — each image_id appears at most once
- **Split strategy:** 80/10/10 stratified by item_id — all views of one item stay in the same split
- **Image processing:** centre-cropped to 224×224, saved as JPEG quality 90
- **Format:** ImageFolder layout (`<split>/<label>/<image_id>.jpg`)

## Statistics

| Class | Train | Val | Test | Total |
|---------|------:|-----:|-----:|-------:|
| bed | 1,263 | 93 | 77 | 1,433 |
| chair | 8,112 | 961 | 905 | 9,978 |
| lamp | 3,499 | 411 | 388 | 4,298 |
| sofa | 2,966 | 333 | 322 | 3,621 |
| storage | 1,766 | 204 | 220 | 2,190 |
| table | 3,093 | 369 | 370 | 3,832 |
| **Total** | **20,699** | **2,371** | **2,282** | **25,352** |

**Note:** Chair is ~7× more frequent than bed. This reflects the original ABO
distribution and is an intentional teaching point about class imbalance.

## Sidecar files

- `metadata.csv` — per-image record: `image_id`, `item_id`, `label`, `split`,
  `is_primary`, `view_index`, `original_height`, `original_width`, `file_path`
- `retrieval_groups.json` — dict keyed by `item_id`; each value has `label`,
  `split`, and `image_ids` (list of all view image_ids for that item). Use this
  for image retrieval evaluation: query on one view, retrieve by item_id.

## Usage in notebooks

```python
from huggingface_hub import hf_hub_download
import tarfile, pathlib

archive = hf_hub_download(
    "marco-willi/abo_furniture",
    "abo_furniture.tar.gz",
    repo_type="dataset",
)
DATA_PATH = pathlib.Path("data")
with tarfile.open(archive) as tar:
    tar.extractall(DATA_PATH)
# Images are now at data/abo/<split>/<label>/<image_id>.jpg
```
