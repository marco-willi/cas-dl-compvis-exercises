"""Step 01 — Download and extract the EuroSAT RGB dataset from Zenodo.

Downloads EuroSAT_RGB.zip (~90 MB) from Zenodo record 7711810 and extracts it
to a raw/ staging directory.

Usage:
    python scripts/prepare_datasets/eurosat/01_download.py
    python scripts/prepare_datasets/eurosat/01_download.py --dest data/raw/eurosat
"""

import argparse
import zipfile
from pathlib import Path

import requests

SOURCE_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEST = REPO_ROOT / "data" / "raw" / "eurosat"


def download(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "EuroSAT_RGB.zip"

    if not archive.exists():
        print(f"Downloading from {SOURCE_URL} ...")
        with requests.get(SOURCE_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with archive.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {pct:.1f}%", end="", flush=True)
        print(f"\nSaved to {archive}")
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

    # EuroSAT_RGB.zip extracts to EuroSAT_RGB/<ClassName>/...
    eurosat_dir = extract_dir / "EuroSAT_RGB"
    if not eurosat_dir.exists():
        # Some versions extract directly without the subdirectory wrapper
        eurosat_dir = extract_dir

    return eurosat_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    args = parser.parse_args()
    eurosat_dir = download(Path(args.dest))
    print(f"\nReady: {eurosat_dir}")
    for cls_dir in sorted(eurosat_dir.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*.jpg")))
            print(f"  {cls_dir.name}: {count} images")
