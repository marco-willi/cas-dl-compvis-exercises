"""Step 01 — Download and extract the Microsoft Cats vs Dogs dataset.

Downloads the source zip from Microsoft, extracts it to a raw/ staging directory,
and removes the two known corrupt images.

Usage:
    python scripts/prepare_datasets/cats_vs_dogs/01_download.py
    python scripts/prepare_datasets/cats_vs_dogs/01_download.py --dest data/raw/cats_vs_dogs
"""

import argparse
import zipfile
from pathlib import Path

import requests

SOURCE_URL = (
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/"
    "kagglecatsanddogs_5340.zip"
)
CORRUPT_FILES = [
    "PetImages/Cat/666.jpg",
    "PetImages/Dog/11702.jpg",
]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEST = REPO_ROOT / "data" / "raw" / "cats_vs_dogs"


def download(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "kagglecatsanddogs_5340.zip"

    if not archive.exists():
        print(f"Downloading from {SOURCE_URL} ...")
        with requests.get(SOURCE_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with archive.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        print(f"Saved to {archive}")
    else:
        print(f"Archive already present: {archive}")

    extract_dir = dest / "extracted"
    if not extract_dir.exists():
        print(f"Extracting to {extract_dir} ...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
        print("Extraction complete.")
    else:
        print(f"Already extracted: {extract_dir}")

    for rel in CORRUPT_FILES:
        bad = extract_dir / rel
        if bad.exists():
            bad.unlink()
            print(f"Removed corrupt file: {bad}")

    return extract_dir / "PetImages"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    args = parser.parse_args()
    pet_images_dir = download(Path(args.dest))
    print(f"\nReady: {pet_images_dir}")
    for cls_dir in sorted(pet_images_dir.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*.jpg")))
            print(f"  {cls_dir.name}: {count} images")
