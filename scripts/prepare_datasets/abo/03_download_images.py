"""Step 3 - Download and resize ABO furniture images.

Reads abo_manifest.json, downloads images from the ABO S3 bucket, applies
center-crop to square and resize to 224x224, and writes ImageFolder layout:

    data/abo/<split>/<label>/<image_id>.jpg

Failed downloads are logged to data/abo/abo_failed.txt.

Usage:
    python scripts/prepare_datasets/abo/03_download_images.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
MANIFEST_PATH = SCRIPT_DIR / "abo_manifest.json"
OUTPUT_DIR = REPO_ROOT / "data" / "abo"
FAILED_LOG_PATH = OUTPUT_DIR / "abo_failed.txt"

ABO_S3_BASE_URL = "https://amazon-berkeley-objects.s3.amazonaws.com"
TARGET_SIZE = 224
MAX_WORKERS = 8
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60
RETRY_BACKOFF_SECONDS = 1.5


def center_crop_resize(image: Image.Image, size: int) -> Image.Image:
    """Crop to the shortest side and resize to ``size`` x ``size``."""
    width, height = image.size
    short_side = min(width, height)
    left = (width - short_side) // 2
    top = (height - short_side) // 2
    right = left + short_side
    bottom = top + short_side
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((size, size), Image.Resampling.LANCZOS)


def build_url(s3_path: str) -> str:
    """Build an absolute URL from a manifest s3_path."""
    return f"{ABO_S3_BASE_URL}/{s3_path.lstrip('/')}"


def download_and_process(entry: dict) -> dict:
    """Download one image, transform, and write it to ImageFolder layout."""
    image_id = entry["image_id"]
    split = entry["split"]
    label = entry["label"]
    url = build_url(entry["s3_path"])

    out_dir = OUTPUT_DIR / split / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_id}.jpg"

    if out_path.exists():
        return {"image_id": image_id, "status": "skipped"}

    last_error = "unknown error"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            with Image.open(BytesIO(response.content)) as image:
                image = image.convert("RGB")
                image = center_crop_resize(image, TARGET_SIZE)
                image.save(out_path, format="JPEG", quality=90)

            return {"image_id": image_id, "status": "ok"}
        except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return {
        "image_id": image_id,
        "status": "failed",
        "url": url,
        "error": last_error,
    }


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}. Run 02_filter_sample.py first."
        )

    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Entries: {len(manifest)}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Workers: {MAX_WORKERS}\n")

    started = time.time()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download_and_process, entry) for entry in manifest]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if idx % 200 == 0 or idx == len(futures):
                elapsed = time.time() - started
                rate = idx / elapsed if elapsed > 0 else 0
                failed_count = sum(1 for r in results if r["status"] == "failed")
                print(
                    f"[{idx}/{len(futures)}] elapsed={elapsed:.0f}s "
                    f"rate={rate:.1f} img/s failed={failed_count}"
                )

    ok_count = sum(1 for r in results if r["status"] == "ok")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    failed = [r for r in results if r["status"] == "failed"]

    if failed:
        with FAILED_LOG_PATH.open("w") as f:
            for item in failed:
                f.write(f"{item['image_id']}\t{item['url']}\t{item['error']}\n")

    total = len(results)
    failure_rate = (len(failed) / total * 100) if total else 0.0
    elapsed = time.time() - started

    print("\nDone.")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"OK: {ok_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {len(failed)}")
    print(f"Failure rate: {failure_rate:.2f}%")
    if failed:
        print(f"Failure log: {FAILED_LOG_PATH}")


if __name__ == "__main__":
    main()
