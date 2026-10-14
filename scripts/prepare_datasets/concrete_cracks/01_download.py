"""Step 01 — Download the Concrete Cracks dataset from Google Drive.

Downloads the original concrete_data.zip (Mendeley / Özgenel dataset) from GDrive,
extracts it to a raw/ staging directory, and prints the internal layout so that
02_restructure.py can be written accordingly.

Usage:
    python scripts/prepare_datasets/concrete_cracks/01_download.py
    python scripts/prepare_datasets/concrete_cracks/01_download.py --dest data/raw/concrete_cracks
"""

import argparse
import zipfile
from pathlib import Path

import gdown

# Original GDrive file ID (concrete_data.zip — Özgenel et al., Mendeley)
GDRIVE_FILE_ID = "1Q-qLQ2RTbpBExsPI4v4B-pBq2K3OrwLb"
ARCHIVE_NAME = "concrete_data.zip"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEST = REPO_ROOT / "data" / "raw" / "concrete_cracks"


def download(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / ARCHIVE_NAME

    if not archive.exists():
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        print(f"Downloading from Google Drive (id={GDRIVE_FILE_ID}) ...")
        gdown.download(url, str(archive), quiet=False, fuzzy=True)
        if not archive.exists():
            raise RuntimeError("gdown reported success but archive is missing")
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

    return extract_dir


def print_layout(extract_dir: Path, max_files: int = 3):
    print(f"\nLayout of {extract_dir}:")
    for p in sorted(extract_dir.rglob("*")):
        rel = p.relative_to(extract_dir)
        depth = len(rel.parts)
        if depth > 3:
            continue
        indent = "  " * (depth - 1)
        if p.is_dir():
            n = sum(1 for _ in p.glob("*") if _.is_file())
            print(f"{indent}{rel.name}/  ({n} files)")
        elif depth <= 2:
            print(f"{indent}{rel.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    args = parser.parse_args()
    extract_dir = download(Path(args.dest))
    print_layout(extract_dir)
    print("\nInspect the layout above before running 02_restructure.py.")
