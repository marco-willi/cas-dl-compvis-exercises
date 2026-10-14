# EuroSAT RGB — Classroom Dataset v1.0

## Summary

The EuroSAT land-use classification dataset (RGB variant), restructured for
classroom use in the CAS Deep Learning — Computer Vision course exercises.
Contains ~27,000 64x64 RGB satellite images across 10 land-use classes,
split into ImageFolder layout.

| Archive | Images | Description |
|---|---:|---|
| `eurosat.tar.gz` | ~27,000 | Full dataset, 70/15/15 stratified split |

## Source

- **Dataset:** EuroSAT — A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification
- **URL:** https://zenodo.org/records/7711810
- **License:** MIT License
- **Attribution:** Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019).
  "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover
  Classification." IEEE Journal of Selected Topics in Applied Earth Observations and
  Remote Sensing. doi: 10.1109/JSTARS.2019.2918242

## Curation

- **Split strategy:** Stratified 70/15/15 (train/val/test), random seed 42
- **Classes (10):** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
  Pasture, PermanentCrop, Residential, River, SeaLake
- **Image resolution:** 64x64 px RGB
- **Format:** JPEG, ImageFolder layout (`<split>/<label>/<filename>.jpg`)
- **Note:** Source images are JPEG; any TIF variants are converted to JPEG during packaging.

## Statistics

| Class | Train | Val | Test | Total |
|----------------------|-------:|------:|------:|-------:|
| AnnualCrop | ~1,890 | ~405 | ~405 | ~2,700 |
| Forest | ~1,890 | ~405 | ~405 | ~2,700 |
| HerbaceousVegetation | ~1,890 | ~405 | ~405 | ~2,700 |
| Highway | ~1,750 | ~375 | ~375 | ~2,500 |
| Industrial | ~1,680 | ~360 | ~360 | ~2,400 |
| Pasture | ~1,330 | ~285 | ~285 | ~1,900 |
| PermanentCrop | ~1,750 | ~375 | ~375 | ~2,500 |
| Residential | ~2,100 | ~450 | ~450 | ~3,000 |
| River | ~1,680 | ~360 | ~360 | ~2,400 |
| SeaLake | ~1,960 | ~420 | ~420 | ~2,800 |
| **Total** | ~17,920| ~3,840| ~3,840| ~25,600 |

*Exact counts vary with random seed; class sizes in the original range from ~1,900 to ~3,000.*

## Sidecar files

- `metadata.csv` — per-image record with `image_id`, `file_name`, `split`, `label`, `width`, `height`

## Suitable exercises

Alternative dataset for **exercises 01, 02, 04, 04_lora, and 05**.
Not suitable for exercise 03 (retrieval — requires multi-view item structure).
Well suited for demonstrating transfer learning on non-photographic imagery and
discussing domain shift from ImageNet pretraining.

## Important: image size

EuroSAT images are **64x64 px**. Exercise notebooks expect 224x224 input for
ImageNet-pretrained models. Apply a resize transform:

```python
transforms.Resize(224)   # or Resize(256) + CenterCrop(224)
```

## Usage in notebooks

```python
dataset_dir = ensure_dataset(
    DATA_PATH, "eurosat.tar.gz", "eurosat",
    gdrive_id="<eurosat-drive-id>",
)
```
