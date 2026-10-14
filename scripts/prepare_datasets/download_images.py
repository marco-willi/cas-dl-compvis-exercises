"""Steps A3/B3 — Download images from a manifest file.

Shared download script for both Kgalagadi and CCT20. Reads a manifest JSON,
downloads images at original resolution, saves in ImageFolder layout, and
writes a metadata.csv sidecar with per-image dimensions.

Usage:
    python scripts/prepare_datasets/download_images.py kgalagadi
    python scripts/prepare_datasets/download_images.py cct20
"""

import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MAX_WORKERS = 8
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

DATASETS = {
    "kgalagadi": {
        "manifest": SCRIPT_DIR / "kgalagadi" / "kgalagadi_manifest.json",
        "output_dir": DATA_DIR / "kgalagadi",
    },
    "cct20": {
        "manifest": SCRIPT_DIR / "cct20" / "cct20_manifest.json",
        "output_dir": DATA_DIR / "cct20",
    },
}


def download_one(entry: dict, output_dir: Path) -> dict:
    """Download a single image. Returns a result dict."""
    split = entry["split"]
    label = entry["label"]
    url = entry["url"]
    file_name = Path(entry["file_name"]).name  # flatten to just the filename

    dest_dir = output_dir / split / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file_name

    if dest_path.exists():
        return {
            "image_id": entry["image_id"],
            "file_name": file_name,
            "split": split,
            "label": label,
            "width": entry.get("width", 0),
            "height": entry.get("height", 0),
            "status": "skipped",
        }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest_path.write_bytes(resp.content)
            return {
                "image_id": entry["image_id"],
                "file_name": file_name,
                "split": split,
                "label": label,
                "width": entry.get("width", 0),
                "height": entry.get("height", 0),
                "status": "ok",
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return {
                    "image_id": entry["image_id"],
                    "file_name": file_name,
                    "split": split,
                    "label": label,
                    "width": 0,
                    "height": 0,
                    "status": f"failed: {e}",
                }


def main(dataset_name: str):
    cfg = DATASETS[dataset_name]
    manifest_path = cfg["manifest"]
    output_dir = cfg["output_dir"]

    with manifest_path.open() as f:
        manifest = json.load(f)

    print(f"Dataset:    {dataset_name}")
    print(f"Manifest:   {manifest_path}")
    print(f"Output:     {output_dir}")
    print(f"Images:     {len(manifest)}")
    print(f"Workers:    {MAX_WORKERS}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failed = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_one, entry, output_dir): entry for entry in manifest
        }
        for done_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if result["status"].startswith("failed"):
                failed.append(result)
            if done_count % 100 == 0 or done_count == len(manifest):
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                print(
                    f"  [{done_count}/{len(manifest)}] "
                    f"{elapsed:.0f}s elapsed, {rate:.1f} img/s, "
                    f"{len(failed)} failed"
                )

    elapsed = time.time() - t0
    ok_count = sum(1 for r in results if r["status"] == "ok")
    skip_count = sum(1 for r in results if r["status"] == "skipped")

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  OK:      {ok_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed:  {len(failed)}")

    # Write failed log
    if failed:
        failed_path = output_dir / "failed_downloads.txt"
        with failed_path.open("w") as f:
            for r in failed:
                f.write(f"{r['image_id']}\t{r['status']}\n")
        print(f"  Failed log: {failed_path}")

    # Write metadata.csv
    csv_path = output_dir / "metadata.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "file_name", "split", "label", "width", "height"],
        )
        writer.writeheader()
        for r in sorted(
            results, key=lambda x: (x["split"], x["label"], x["file_name"])
        ):
            if r["status"].startswith("failed"):
                continue
            writer.writerow(
                {
                    "image_id": r["image_id"],
                    "file_name": r["file_name"],
                    "split": r["split"],
                    "label": r["label"],
                    "width": r["width"],
                    "height": r["height"],
                }
            )
    print(f"  Metadata:  {csv_path}")

    # Verify with quick count
    print("\nDirectory structure:")
    for split_dir in sorted(output_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            n = len(list(label_dir.glob("*")))
            print(f"  {split_dir.name}/{label_dir.name}: {n} files")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in DATASETS:
        print(f"Usage: python {sys.argv[0]} <{'|'.join(DATASETS)}>")
        sys.exit(1)
    main(sys.argv[1])
