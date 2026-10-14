"""Step 10 — Download and resize ser_sampled images.

Reads ser_sampled_manifest.json, downloads each image from Azure blob storage,
resizes to max 1024 px on the longer side (aspect-ratio preserving, Lanczos),
and saves as JPEG quality 92.

Writes:
    data/ser/ser_sampled/<split>/<label>/<stem>.jpg
    data/ser/ser_sampled/metadata.csv
    data/ser/ser_sampled/ser_sampled_failed.txt  (only if failures occur)

Usage:
    python scripts/prepare_datasets/ser/10_download_sampled.py
"""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
MANIFEST_PATH = SCRIPT_DIR / "ser_sampled_manifest.json"
OUTPUT_DIR = DATA_DIR / "ser" / "ser_sampled"

MAX_SIZE = 1024
JPEG_QUALITY = 92
MAX_WORKERS = 8
MAX_RETRIES = 3
RETRY_DELAY = 2


def process_one(entry: dict, output_dir: Path) -> dict:
    split = entry["split"]
    label = entry["label"]
    url = entry["url"]
    image_id = entry["image_id"]
    stem = Path(image_id).stem

    dest_dir = output_dir / split / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{stem}.jpg"

    if dest_path.exists():
        try:
            with Image.open(dest_path) as im:
                w, h = im.size
        except Exception:
            w, h = 0, 0
        return {
            "image_id": image_id,
            "file_name": entry["file_name"],
            "split": split,
            "label": label,
            "width": w,
            "height": h,
            "md_bbox": json.dumps(entry.get("md_bbox")),
            "md_conf": entry.get("md_conf", 0),
            "status": "skipped",
        }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_SIZE:
                if w >= h:
                    new_w, new_h = MAX_SIZE, round(h * MAX_SIZE / w)
                else:
                    new_w, new_h = round(w * MAX_SIZE / h), MAX_SIZE
                img = img.resize((new_w, new_h), Image.LANCZOS)
                w, h = new_w, new_h
            img.save(dest_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
            return {
                "image_id": image_id,
                "file_name": entry["file_name"],
                "split": split,
                "label": label,
                "width": w,
                "height": h,
                "md_bbox": json.dumps(entry.get("md_bbox")),
                "md_conf": entry.get("md_conf", 0),
                "status": "ok",
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return {
                    "image_id": image_id,
                    "file_name": entry["file_name"],
                    "split": split,
                    "label": label,
                    "width": 0,
                    "height": 0,
                    "md_bbox": json.dumps(entry.get("md_bbox")),
                    "md_conf": entry.get("md_conf", 0),
                    "status": f"failed: {e}",
                }


with MANIFEST_PATH.open() as f:
    manifest = json.load(f)

print(f"Manifest:  {MANIFEST_PATH}")
print(f"Output:    {OUTPUT_DIR}")
print(f"Images:    {len(manifest)}")
print(f"Workers:   {MAX_WORKERS}")
print(f"Max size:  {MAX_SIZE}px")
print()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
results = []
failed = []
t0 = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {pool.submit(process_one, entry, OUTPUT_DIR): entry for entry in manifest}
    for done_count, future in enumerate(as_completed(futures), start=1):
        result = future.result()
        results.append(result)
        if result["status"].startswith("failed"):
            failed.append(result)
        if done_count % 100 == 0 or done_count == len(manifest):
            elapsed = time.time() - t0
            rate = done_count / elapsed if elapsed > 0 else 0
            print(
                f"  [{done_count:>4}/{len(manifest)}] "
                f"{elapsed:.0f}s elapsed  {rate:.1f} img/s  {len(failed)} failed",
                end="\r",
            )

elapsed = time.time() - t0
ok_count = sum(1 for r in results if r["status"] == "ok")
skip_count = sum(1 for r in results if r["status"] == "skipped")
print(f"\n\nDone in {elapsed:.0f}s")
print(f"  OK:      {ok_count}")
print(f"  Skipped: {skip_count}")
print(f"  Failed:  {len(failed)}")

if failed:
    fail_path = OUTPUT_DIR / "ser_sampled_failed.txt"
    with fail_path.open("w") as f:
        for r in failed:
            f.write(f"{r['image_id']}\t{r['file_name']}\t{r['status']}\n")
    print(f"  Failures logged to: {fail_path}")
    fail_rate = len(failed) / len(manifest)
    if fail_rate >= 0.05:
        print(f"  WARNING: failure rate {fail_rate:.1%} >= 5%")

csv_path = OUTPUT_DIR / "metadata.csv"
fieldnames = [
    "image_id",
    "file_name",
    "split",
    "label",
    "width",
    "height",
    "md_bbox",
    "md_conf",
]
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in sorted(results, key=lambda x: (x["split"], x["label"], x["image_id"])):
        if not r["status"].startswith("failed"):
            writer.writerow({k: r[k] for k in fieldnames})
print(f"  Metadata: {csv_path}")

print("\nDirectory structure:")
for split_dir in sorted(OUTPUT_DIR.iterdir()):
    if not split_dir.is_dir():
        continue
    for label_dir in sorted(split_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        n = len(list(label_dir.glob("*.jpg")))
        print(f"  {split_dir.name}/{label_dir.name}: {n} files")
