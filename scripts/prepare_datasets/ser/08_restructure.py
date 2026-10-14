"""Step 8 — Restructure SER data under a unified root.

Moves the current flat layout:
    data/ser/{train,val,test}/          → data/ser/ser_balanced/{train,val,test}/
    data/ser/{metadata.csv,README.md}   → data/ser/ser_balanced/
    data/ser_cropped/{train,val,test}/  → data/ser/ser_balanced_cropped/{train,val,test}/
    data/ser_cropped/{metadata_cropped.csv,README.md}
                                        → data/ser/ser_balanced_cropped/

After the move, data/ser_cropped/ is removed and data/ser/ contains only the
four named sub-dataset directories.

Usage:
    python scripts/prepare_datasets/ser/08_restructure.py
"""

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

SER_ROOT = DATA_DIR / "ser"
OLD_CROPPED = DATA_DIR / "ser_cropped"

BALANCED = SER_ROOT / "ser_balanced"
BALANCED_CROPPED = SER_ROOT / "ser_balanced_cropped"

SPLIT_DIRS = ["train", "val", "test"]


def move_splits(src: Path, dst: Path, label: str):
    dst.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_DIRS:
        src_split = src / split
        dst_split = dst / split
        if src_split.exists():
            shutil.move(str(src_split), str(dst_split))
            print(
                f"  Moved {src_split.relative_to(REPO_ROOT)} → {dst_split.relative_to(REPO_ROOT)}"
            )
        else:
            print(f"  SKIP (not found): {src_split.relative_to(REPO_ROOT)}")


def move_file(src: Path, dst_dir: Path):
    if src.exists():
        shutil.move(str(src), str(dst_dir / src.name))
        print(f"  Moved {src.name} → {dst_dir.relative_to(REPO_ROOT)}/")
    else:
        print(f"  SKIP (not found): {src.name}")


# ---------------------------------------------------------------------------
# Guard: don't re-run if already restructured
# ---------------------------------------------------------------------------
if BALANCED.exists():
    print("ser_balanced/ already exists — looks like restructure already ran.")
    print("Delete ser_balanced/ first if you want to re-run.")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# 1. ser_balanced — move splits + sidecar files from data/ser/
# ---------------------------------------------------------------------------
print("\n=== Creating ser_balanced/ ===")
move_splits(SER_ROOT, BALANCED, "ser_balanced")
for fname in ["metadata.csv", "README.md", "ser_failed.txt"]:
    move_file(SER_ROOT / fname, BALANCED)

# ---------------------------------------------------------------------------
# 2. ser_balanced_cropped — move from data/ser_cropped/
# ---------------------------------------------------------------------------
print("\n=== Creating ser_balanced_cropped/ ===")
if OLD_CROPPED.exists():
    move_splits(OLD_CROPPED, BALANCED_CROPPED, "ser_balanced_cropped")
    for fname in ["metadata_cropped.csv", "README.md", "crop_errors.txt"]:
        move_file(OLD_CROPPED / fname, BALANCED_CROPPED)
    # Remove now-empty old dir
    remaining = list(OLD_CROPPED.iterdir())
    if remaining:
        print(f"  WARNING: {len(remaining)} unexpected files remain in ser_cropped/:")
        for p in remaining:
            print(f"    {p.name}")
    else:
        OLD_CROPPED.rmdir()
        print(f"  Removed {OLD_CROPPED.relative_to(REPO_ROOT)}/")
else:
    print(f"  SKIP: {OLD_CROPPED} not found")

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
print("\n=== Verification ===")
for sub in [BALANCED, BALANCED_CROPPED]:
    if sub.exists():
        n_imgs = len(list(sub.rglob("*.jpg")))
        print(f"  {sub.name}/  —  {n_imgs} images")
    else:
        print(f"  MISSING: {sub.name}/")

stray = [
    p
    for p in SER_ROOT.iterdir()
    if p.name
    not in {
        "ser_balanced",
        "ser_balanced_cropped",
        "ser_sampled",
        "ser_sampled_cropped",
    }
]
if stray:
    print(f"\n  WARNING: unexpected items in data/ser/: {[p.name for p in stray]}")
else:
    print("\n  data/ser/ root is clean.")

print("\nDone.")
