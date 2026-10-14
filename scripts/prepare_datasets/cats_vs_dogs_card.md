# Microsoft Cats vs Dogs — Classroom Dataset v1.0

## Summary

The full Microsoft Cats vs Dogs dataset, restructured for classroom use in the
CAS Deep Learning — Computer Vision course exercises. Contains ~25,000 colour
photos of cats and dogs split into ImageFolder layout.

| Archive | Images | Description |
|---|---:|---|
| `cats_vs_dogs.tar.gz` | ~25,000 | Full dataset, 70/15/15 stratified split |

## Source

- **Dataset:** Microsoft Cats vs Dogs (Kaggle Cats and Dogs Dataset)
- **URL:** https://www.microsoft.com/en-us/download/details.aspx?id=54765
- **License:** Microsoft Terms of Use (non-commercial, educational use)
- **Attribution:** Microsoft Corporation

## Curation

- **Corrupt files removed:** `PetImages/Cat/666.jpg`, `PetImages/Dog/11702.jpg`
- **Split strategy:** Stratified 70/15/15 (train/val/test), random seed 42
- **Classes:** Cat, Dog
- **Image resolution:** Original (variable, typically 150-500 px on the longer side)
- **Format:** JPEG, ImageFolder layout (`<split>/<label>/<filename>.jpg`)

## Statistics

| Class | Train | Val | Test | Total |
|-----------|-------:|-------:|-------:|-------:|
| Cat | ~8,750 | ~1,875 | ~1,875 | ~12,500 |
| Dog | ~8,750 | ~1,875 | ~1,875 | ~12,500 |
| **Total** | ~17,500 | ~3,750 | ~3,750 | ~25,000 |

## Sidecar files

- `metadata.csv` — per-image record with `image_id`, `file_name`, `split`, `label`, `width`, `height`

## Suitable exercises

Alternative dataset for **exercises 01, 02, 04, 04_lora, and 05**.
Not suitable for exercise 03 (retrieval — requires multi-view item structure).

## Usage in notebooks

```python
dataset_dir = ensure_dataset(
    DATA_PATH, "cats_vs_dogs.tar.gz", "cats_vs_dogs",
    gdrive_id="<cats_vs_dogs-drive-id>",
)
```

Apply `transforms.Resize(256)` + `CenterCrop(224)` — images are variable size.
