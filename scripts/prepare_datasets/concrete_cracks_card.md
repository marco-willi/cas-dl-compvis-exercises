# Concrete Cracks — Classroom Dataset v1.0

## Summary

The Ozgenel et al. Concrete Crack Images dataset, restructured for classroom use
in the CAS Deep Learning — Computer Vision course exercises. Contains ~40,000
227x227 JPEG images of concrete surfaces labelled as cracked (Positive) or
intact (Negative), split into ImageFolder layout.

| Archive | Images | Description |
|---|---:|---|
| `concrete_cracks.tar.gz` | ~40,000 | Full dataset, 70/15/15 stratified split |

## Source

- **Dataset:** Concrete Crack Images for Classification
- **URL:** https://data.mendeley.com/datasets/5y9wdsg2zt/2
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Attribution:** Ozgenel, Caglar Firat (2019). "Concrete Crack Images for Classification."
  Mendeley Data, V2. doi: 10.17632/5y9wdsg2zt.2

## Curation

- **Split strategy:** Stratified 70/15/15 (train/val/test), random seed 42
- **Classes:** Negative (no crack), Positive (crack present)
- **Image resolution:** 227x227 px
- **Format:** JPEG, ImageFolder layout (`<split>/<label>/<filename>.jpg`)

## Statistics

| Class | Train | Val | Test | Total |
|-----------|--------:|-------:|-------:|--------:|
| Negative | ~14,000 | ~3,000 | ~3,000 | ~20,000 |
| Positive | ~14,000 | ~3,000 | ~3,000 | ~20,000 |
| **Total** | ~28,000 | ~6,000 | ~6,000 | ~40,000 |

## Sidecar files

- `metadata.csv` — per-image record with `image_id`, `file_name`, `split`, `label`, `width`, `height`

## Suitable exercises

Alternative dataset for **exercises 01, 02, 04, 04_lora, and 05**.
Not suitable for exercise 03 (retrieval — requires multi-view item structure).
Well suited for demonstrating binary classification and defect detection.

## Usage in notebooks

```python
dataset_dir = ensure_dataset(
    DATA_PATH, "concrete_cracks.tar.gz", "concrete_cracks",
    gdrive_id="<concrete_cracks-drive-id>",
)
```

Images are 227x227 px — `transforms.Resize(224)` or `CenterCrop(224)` suffices.
