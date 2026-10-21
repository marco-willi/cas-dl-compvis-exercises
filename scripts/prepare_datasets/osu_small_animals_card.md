---
license: cdla-permissive-1.0
task_categories:
  - image-classification
tags:
  - camera-traps
  - wildlife
  - ecology
pretty_name: Ohio Small Animals — Species Classroom Subset
size_categories:
  - 10K<n<100K
---

# Ohio Small Animals — Species Classroom Subset v1.0

## Summary

A 16-species subset of the LILA Ohio Small Animals camera trap dataset for use in
the CAS Deep Learning — Computer Vision course. The cameras look straight down into
a bucket that snakes, small mammals, lizards and ground birds are guided into by
drift fences. The train, validation and test images come from different sites, so
the test score measures how well a model works at sites it has not seen.

## Source

- **Dataset:** Ohio Small Animals (AHDriFT camera traps)
- **URL:** https://lila.science/datasets/ohio-small-animals/
- **License:** Community Data License Agreement — Permissive variant 1.0
- **Attribution:** Balasubramaniam S., "Optimized Classification in Camera Trap
  Images: An Approach with Smart Camera Traps, Machine Learning, and Human
  Inference", Master's thesis, The Ohio State University, 2024

## License

This subset is published under the
[Community Data License Agreement — Permissive, Version 1.0](https://cdla.dev/permissive-1-0/),
the same license as the source dataset. If you use it, please cite the source above.

**Notice of changes:** these files are modified from the original Ohio Small Animals
data. Images were selected, renamed, resized to max 1024 px and re-encoded as JPEG,
and the labels and metadata were rewritten into the `metadata.csv` files (see
Curation below).

## Curation

- **Selected classes:** every species with at least 500 images at 10 or more sites.
  Left out: `empty` (only 5 sites, all with one camera setup, so the camera would give
  away the label) and `invertebrate` (a catch-all of mostly tiny insects and spiders).
- **Split strategy:** by site — all cameras of one deployment (e.g. KPC1/KPC2/KPC3,
  or the N/S pair of a 2022 grid point such as KILC4N/KILC4S) go to the same split.
  Sites were assigned so that each species has about 70/15/15 of its sequences in
  train/val/test, with at least a few sequences in val and test.
- **Sampling:** at most 800 train / 200 val / 200 test images per species. Images
  are taken one per sequence first, then a second and third evenly spaced frame
  from the same sequence if needed (at most 3 per sequence). Use `seq_id` to group
  frames of one sequence.
- **Image resolution:** resized from 2048 × 1440 to max 1024 px on the longest side
  (1024 × 720), JPEG quality 90
- **Format:** JPEG, ImageFolder layout (`<split>/<label>/<filename>.jpg`)

## Statistics

Images (sequences) per split:

| Class | Train | Val | Test |
|-------------------------|-----------:|-----------:|-----------:|
| eastern_gartersnake | 800 (800) | 200 (200) | 200 (200) |
| song_sparrow | 800 (800) | 64 (23) | 200 (200) |
| meadow_vole | 800 (800) | 200 (104) | 200 (200) |
| white-footed_mouse | 800 (800) | 200 (200) | 200 (200) |
| northern_house_wren | 800 (529) | 188 (70) | 104 (38) |
| common_five-lined_skink | 800 (545) | 200 (200) | 43 (15) |
| masked_shrew | 800 (800) | 200 (158) | 200 (200) |
| eastern_cottontail | 780 (311) | 60 (23) | 130 (58) |
| long-tailed_weasel | 395 (152) | 82 (31) | 112 (42) |
| woodland_jumping_mouse | 682 (300) | 188 (116) | 51 (24) |
| plains_gartersnake | 519 (181) | 117 (42) | 97 (36) |
| eastern_massasauga | 349 (120) | 64 (23) | 98 (33) |
| virginia_opossum | 133 (46) | 75 (25) | 43 (15) |
| common_yellowthroat | 145 (59) | 60 (21) | 49 (19) |
| n.\_short-tailed_shrew | 359 (149) | 52 (23) | 49 (20) |
| dekay's_brownsnake | 224 (79) | 50 (18) | 46 (16) |
| **Total** | **9186** | **2000** | **1822** |

Sites: 43 train, 11 val, 8 test.

**Notes:**

- The classes are imbalanced (133 to 800 training images).
- Most species occur at only 2–3 of the val/test sites, so per-class val/test scores
  rest on few sites and are noisy for the small classes.
- Many images are night-time infrared (grayscale).
- Labels are per sequence in the source, so some frames show the animal only partly.

## Sidecar files

- `metadata.csv` — one row per image with `image_id`, `file_name` (relative to the
  dataset folder), `split`, `label`, `site`, `location` (camera), `seq_id`,
  `frame_num`, `datetime`, `source_file`, `width`, `height`
- `<split>/metadata.csv` — the same rows with `file_name` relative to the split
  folder; these are what Hugging Face `datasets` reads

## Usage in notebooks

```python
from huggingface_hub import hf_hub_download
import tarfile

archive = hf_hub_download(
    "marco-willi/camera-trap-osu-small-animals",
    "osu_small_animals.tar.gz",
    repo_type="dataset",
)
with tarfile.open(archive) as tar:
    tar.extractall(DATA_PATH)
```

With Hugging Face `datasets` (the `val` folder becomes the `validation` split):

```python
from datasets import load_dataset

ds = load_dataset("imagefolder", data_dir=DATA_PATH / "osu_small_animals")
ds = ds.class_encode_column("label")  # species name -> ClassLabel
```

With torchvision, `ImageFolder(DATA_PATH / "osu_small_animals" / "train")` works
directly; the `metadata.csv` file in each split folder is ignored.
