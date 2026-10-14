"""Step 11 — Crop ser_sampled images using MegaDetector bounding boxes.

Reads ser_sampled_manifest.json, crops each downloaded image with 10% padding,
resizes to max 1024 px, saves as JPEG quality 92.

Writes:
    data/ser/ser_sampled_cropped/<split>/<label>/<stem>.jpg
    data/ser/ser_sampled_cropped/metadata_cropped.csv

Usage:
    python scripts/prepare_datasets/ser/11_crop_sampled.py
"""

import csv
import json
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
MANIFEST_PATH = SCRIPT_DIR / "ser_sampled_manifest.json"
SRC_DIR = DATA_DIR / "ser" / "ser_sampled"
OUT_DIR = DATA_DIR / "ser" / "ser_sampled_cropped"

MAX_SIZE = 1024
JPEG_QUALITY = 92
PADDING_FRAC = 0.10


def crop_with_padding(img: Image.Image, bbox_norm: list[float]) -> Image.Image:
    W, H = img.size
    x, y, bw, bh = bbox_norm
    x1, y1 = x * W, y * H
    x2, y2 = (x + bw) * W, (y + bh) * H
    pad_x = PADDING_FRAC * (x2 - x1)
    pad_y = PADDING_FRAC * (y2 - y1)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x)
    y2 = min(H, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


def resize_if_needed(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_SIZE:
        return img
    if w >= h:
        new_w, new_h = MAX_SIZE, round(h * MAX_SIZE / w)
    else:
        new_w, new_h = round(w * MAX_SIZE / h), MAX_SIZE
    return img.resize((new_w, new_h), Image.LANCZOS)


with MANIFEST_PATH.open() as f:
    manifest = json.load(f)

croppable = [e for e in manifest if e.get("md_bbox") is not None]
skipped_empty = len(manifest) - len(croppable)

print(f"Manifest total:    {len(manifest)}")
print(f"  With bbox:       {len(croppable)}")
print(f"  Skipped (empty): {skipped_empty}")
print(f"Output dir:        {OUT_DIR}")
print()

OUT_DIR.mkdir(parents=True, exist_ok=True)
results = []
errors = []

for i, entry in enumerate(croppable, start=1):
    image_id = entry["image_id"]
    split = entry["split"]
    label = entry["label"]
    md_bbox = entry["md_bbox"]
    stem = Path(image_id).stem

    src_path = SRC_DIR / split / label / f"{stem}.jpg"
    dest_dir = OUT_DIR / split / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{stem}.jpg"

    if dest_path.exists():
        try:
            with Image.open(dest_path) as im:
                w, h = im.size
            results.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "label": label,
                    "width": w,
                    "height": h,
                    "status": "skipped",
                }
            )
        except Exception:
            pass
        if i % 200 == 0:
            print(f"  [{i}/{len(croppable)}] processing...", end="\r")
        continue

    if not src_path.exists():
        errors.append(f"{image_id}: source not found at {src_path}")
        if i % 200 == 0:
            print(f"  [{i}/{len(croppable)}] processing...", end="\r")
        continue

    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            cropped = crop_with_padding(img, md_bbox)
            cropped = resize_if_needed(cropped)
            w, h = cropped.size
            if w == 0 or h == 0:
                raise ValueError(f"Degenerate crop: {w}x{h}")
            cropped.save(dest_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
        results.append(
            {
                "image_id": image_id,
                "split": split,
                "label": label,
                "width": w,
                "height": h,
                "status": "ok",
            }
        )
    except Exception as e:
        errors.append(f"{image_id}: {e}")

    if i % 200 == 0 or i == len(croppable):
        print(f"  [{i:>4}/{len(croppable)}] {len(errors)} errors so far", end="\r")

print("\n\nDone.")
ok_count = sum(1 for r in results if r["status"] == "ok")
skip_count = sum(1 for r in results if r["status"] == "skipped")
print(f"  OK:      {ok_count}")
print(f"  Skipped: {skip_count}")
print(f"  Errors:  {len(errors)}")

if errors:
    err_path = OUT_DIR / "crop_errors.txt"
    with err_path.open("w") as f:
        f.write("\n".join(errors))
    print(f"  Errors logged to: {err_path}")

csv_path = OUT_DIR / "metadata_cropped.csv"
fieldnames = ["image_id", "split", "label", "width", "height"]
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in sorted(results, key=lambda x: (x["split"], x["label"], x["image_id"])):
        if r["status"] in ("ok", "skipped"):
            writer.writerow({k: r[k] for k in fieldnames})
print(f"  Metadata: {csv_path}")

print("\nDirectory structure (ser_sampled_cropped):")
for split_dir in sorted(OUT_DIR.iterdir()):
    if not split_dir.is_dir():
        continue
    for label_dir in sorted(split_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        n = len(list(label_dir.glob("*.jpg")))
        print(f"  {split_dir.name}/{label_dir.name}: {n} files")
