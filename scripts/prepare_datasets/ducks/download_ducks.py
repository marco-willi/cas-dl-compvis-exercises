"""Download the non-empty images of the LILA "Duck Pictures in Wetlands" dataset.

Builds a manifest from the COCO Camera Traps labels, downloads only the images
with at least one usable category, downscales them on the fly (originals are
never stored) and verifies the result.

Dataset page: https://lila.science/datasets/duck-pictures-in-wetlands/
Licence: CDLA-Permissive 1.0

Usage:
    python scripts/prepare_datasets/ducks/download_ducks.py --limit 50
    python scripts/prepare_datasets/ducks/download_ducks.py
    python scripts/prepare_datasets/ducks/download_ducks.py --out /other/repo/demos/data/ducks
"""

import argparse
import csv
import io
import json
import random
import threading
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFile, ImageOps

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "ducks"
LABELS_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "duck-pictures-in-wetlands/duck_pictures_in_wetlands.json.zip"
)
IMAGE_BASE_URL = (
    "https://storage.googleapis.com/public-datasets-lila/duck-pictures-in-wetlands/"
)

# Expected image counts per kept category (from the label JSON).
KEEP_CATEGORIES = {
    "mallard_adult": 3973,
    "goldeneye_adult": 2205,
    "teal_adult": 1880,
    "crane": 904,
    "mammal": 396,
    "mallard_duckling": 394,
    "goldeneye_duckling": 304,
    "swan": 239,
    "teal_duckling": 187,
}
EMPTY = "empty"

MAX_SIDE = 1024
JPEG_QUALITY = 90
MAX_WORKERS = 16
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, doubled after each attempt
TIMEOUT = 60
SEED = 42

METADATA_FIELDS = ["file_name", "location", "labels", "n_labels", "local_path"]

_thread_local = threading.local()


class DecodeLock:
    """Strict decodes run concurrently; a lenient decode runs alone.

    Pillow's lenient mode is a process-wide flag, so a strict decode must not overlap with it.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._strict = 0
        self._lenient = False

    def acquire(self, lenient: bool) -> None:
        with self._cond:
            self._cond.wait_for(
                lambda: not self._lenient and not (lenient and self._strict)
            )
            if lenient:
                self._lenient = True
                ImageFile.LOAD_TRUNCATED_IMAGES = True
            else:
                self._strict += 1

    def release(self, lenient: bool) -> None:
        with self._cond:
            if lenient:
                self._lenient = False
                ImageFile.LOAD_TRUNCATED_IMAGES = False
            else:
                self._strict -= 1
            self._cond.notify_all()


_decode_lock = DecodeLock()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


# ---------------------------------------------------------------------------
# Step 1: manifest
# ---------------------------------------------------------------------------
def load_labels(raw_dir: Path) -> dict:
    """Download (once) and load the COCO Camera Traps label JSON."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "duck_pictures_in_wetlands.json.zip"
    if not zip_path.exists():
        print(f"Downloading labels: {LABELS_URL}")
        resp = requests.get(LABELS_URL, timeout=TIMEOUT)
        resp.raise_for_status()
        tmp_path = zip_path.with_name(zip_path.name + ".part")
        tmp_path.write_bytes(resp.content)
        tmp_path.replace(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        json_names = [n for n in zf.namelist() if n.endswith(".json")]
        json_path = raw_dir / Path(json_names[0]).name
        if not json_path.exists():
            json_path.write_bytes(zf.read(json_names[0]))
    with json_path.open() as f:
        return json.load(f)


def build_manifest(out_dir: Path) -> list[dict]:
    """Write metadata.csv with one row per kept image and return the rows."""
    labels = load_labels(out_dir / "raw")
    categories = {c["id"]: c["name"] for c in labels["categories"]}
    print(
        f"Label JSON: {len(labels['images']):,} images, "
        f"{len({im['location'] for im in labels['images']})} locations, "
        f"{len(categories)} categories"
    )

    labels_per_image = defaultdict(set)
    for ann in labels["annotations"]:
        labels_per_image[ann["image_id"]].add(categories[ann["category_id"]])

    rows = []
    for im in labels["images"]:
        names = labels_per_image.get(im["id"], set())
        if EMPTY in names or not names & KEEP_CATEGORIES.keys():
            continue
        rows.append(
            {
                "file_name": im["file_name"],
                "location": im["location"],
                "labels": "|".join(sorted(names)),
                "n_labels": len(names),
                "local_path": f"images/{im['file_name']}",
            }
        )
    rows.sort(key=lambda r: r["file_name"])

    counts = Counter(name for r in rows for name in r["labels"].split("|"))
    print(f"\nKept images: {len(rows):,}")
    print(f"{'category':<24}{'kept':>8}{'expected':>10}")
    for name, expected in KEEP_CATEGORIES.items():
        flag = "" if counts[name] == expected else "  <-- MISMATCH"
        print(f"{name:<24}{counts[name]:>8}{expected:>10}{flag}")
    for name in sorted(counts.keys() - KEEP_CATEGORIES.keys()):
        print(f"{name:<24}{counts[name]:>8}{'-':>10}  (co-label only)")

    write_metadata(out_dir, rows, METADATA_FIELDS)
    return rows


def write_metadata(out_dir: Path, rows: list[dict], fields: list[str]) -> None:
    path = out_dir / "metadata.csv"
    tmp_path = path.with_name(path.name + ".part")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Step 2: download and downscale
# ---------------------------------------------------------------------------
def select_limited(rows: list[dict], limit: int) -> list[dict]:
    """Pick `limit` rows round-robin over the categories so a test run covers all of them."""
    rng = random.Random(SEED)
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["labels"].split("|")[0]].append(row)
    for group in by_category.values():
        rng.shuffle(group)

    selected = []
    groups = [by_category[name] for name in sorted(by_category)]
    while len(selected) < limit and any(groups):
        for group in groups:
            if group and len(selected) < limit:
                selected.append(group.pop())
    return selected


def decode(content: bytes, max_side: int) -> tuple[Image.Image, bool]:
    """Decode a JPEG at reduced size. Returns the image and whether lenient decoding was needed.

    Many originals are complete downloads whose JPEG stream Pillow rejects as broken
    (the cameras append an "AE Info" text block). They decode fully in lenient mode.
    """

    def load(lenient: bool) -> Image.Image:
        _decode_lock.acquire(lenient)
        try:
            img = Image.open(io.BytesIO(content))
            img.draft(
                "RGB", (max_side, max_side)
            )  # decode at reduced size, never below max_side
            img.load()
            return img
        finally:
            _decode_lock.release(lenient)

    try:
        return load(lenient=False), False
    except OSError:
        return load(lenient=True), True


def download_one(row: dict, out_dir: Path, max_side: int) -> tuple[str | None, bool]:
    """Fetch, downscale and save one image. Returns (error message or None, lenient decode)."""
    dest_path = out_dir / row["local_path"]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    url = IMAGE_BASE_URL + quote(row["file_name"], safe="/")

    error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = get_session().get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            img, lenient = decode(resp.content, max_side)
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail(
                (max_side, max_side), Image.Resampling.LANCZOS
            )  # never upscales
            img.save(tmp_path, format="JPEG", quality=JPEG_QUALITY)
            tmp_path.replace(dest_path)
            return None, lenient
        except Exception as e:  # any failure is retried, then recorded
            error = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * 2**attempt)
    tmp_path.unlink(missing_ok=True)
    return error, False


def download_all(rows: list[dict], out_dir: Path, workers: int, max_side: int) -> None:
    todo = [r for r in rows if not (out_dir / r["local_path"]).exists()]
    print(
        f"\nDownload: {len(rows):,} selected, {len(rows) - len(todo):,} already on disk, {len(todo):,} to fetch"
    )

    failed, lenient_names = [], []
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, r, out_dir, max_side): r for r in todo}
        for i, future in enumerate(as_completed(futures), 1):
            error, lenient = future.result()
            if error:
                failed.append(
                    {"file_name": futures[future]["file_name"], "error": error}
                )
            elif lenient:
                lenient_names.append(futures[future]["file_name"])
            if i % 200 == 0 or i == len(todo):
                rate = i / (time.time() - start)
                print(
                    f"  {i:,}/{len(todo):,}  failed={len(failed)}  {rate:.1f} img/s",
                    flush=True,
                )

    with (out_dir / "failed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "error"])
        writer.writeheader()
        writer.writerows(failed)
    # Appended across runs: these files are fetched only once.
    lenient_path = out_dir / "lenient_decodes.csv"
    is_new = not lenient_path.exists()
    with lenient_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["file_name"])
        writer.writerows([name] for name in sorted(lenient_names))
    print(
        f"Fetched {len(todo) - len(failed):,} ({len(lenient_names):,} decoded leniently, see lenient_decodes.csv), "
        f"failed {len(failed):,} (see failed.csv)"
    )


# ---------------------------------------------------------------------------
# Step 3: verify
# ---------------------------------------------------------------------------
def verify(rows: list[dict], out_dir: Path, max_side: int) -> None:
    """Check every saved file, add width/height to metadata.csv and print a summary."""
    unreadable, too_large, missing, total_bytes = [], [], 0, 0
    present = []
    for row in rows:
        path = out_dir / row["local_path"]
        row["width"] = row["height"] = ""
        if not path.exists():
            missing += 1
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                row["width"], row["height"] = img.size
        except Exception:
            unreadable.append(row["file_name"])
            continue
        if max(row["width"], row["height"]) > max_side:
            too_large.append(row["file_name"])
        total_bytes += path.stat().st_size
        present.append(row)

    write_metadata(out_dir, rows, [*METADATA_FIELDS, "width", "height"])

    print(
        f"\nVerify: {len(present):,} readable, {missing:,} not on disk, {len(unreadable)} unreadable"
    )
    print(f"Longest side > {max_side}px: {len(too_large)}")
    print(f"Size on disk: {total_bytes / 1e9:.2f} GB")
    for name in unreadable[:20]:
        print(f"  unreadable: {name}")

    per_category = Counter(name for r in present for name in r["labels"].split("|"))
    print("\nPer category (on disk):")
    for name, n in per_category.most_common():
        print(f"  {name:<24}{n:>7}")
    per_location = Counter(r["location"] for r in present)
    print(f"\nPer location (on disk, {len(per_location)} locations):")
    for name, n in sorted(per_location.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<24}{n:>7}")

    make_contact_sheet(present, out_dir)


def make_contact_sheet(
    rows: list[dict], out_dir: Path, per_category: int = 4, cell: int = 512
) -> None:
    """Save a grid with random single-label images per category (one row per category)."""
    rng = random.Random(SEED)
    cell_h, header = cell * 3 // 4, 24
    names = [
        n for n in KEEP_CATEGORIES if any(n in r["labels"].split("|") for r in rows)
    ]
    if not names:
        return
    sheet = Image.new(
        "RGB", (cell * per_category, (cell_h + header) * len(names)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(names):
        candidates = [r for r in rows if r["labels"] == name] or [
            r for r in rows if name in r["labels"].split("|")
        ]
        top = i * (cell_h + header)
        draw.text((6, top + 6), f"{name}  (n={KEEP_CATEGORIES[name]})", fill="black")
        for j, row in enumerate(
            rng.sample(candidates, min(per_category, len(candidates)))
        ):
            with Image.open(out_dir / row["local_path"]) as img:
                thumb = ImageOps.pad(img.convert("RGB"), (cell, cell_h), color="black")
            sheet.paste(thumb, (j * cell, top + header))
    path = out_dir / "contact_sheet.jpg"
    sheet.save(path, quality=90)
    print(f"\nContact sheet: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="output directory (default: data/ducks)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="download only N images, spread over the categories",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-side", type=int, default=MAX_SIDE)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = build_manifest(args.out)
    selected = select_limited(rows, args.limit) if args.limit else rows
    download_all(selected, args.out, args.workers, args.max_side)
    verify(rows, args.out, args.max_side)


if __name__ == "__main__":
    main()
