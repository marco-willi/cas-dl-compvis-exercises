"""Prepare a species classification subset of LILA "Ohio Small Animals".

Keeps the species with enough images and sites, splits by site (all cameras of one
deployment array go to the same split), samples a capped number of images per species
and split with at most a few frames per sequence, downloads them, downscales them to
at most 1024 px and writes an ImageFolder layout that
`datasets.load_dataset("imagefolder", ...)` can read.

Output (data/osu_small_animals/):
    <split>/<species>/<image>.jpg
    metadata.csv                    one row per image, file_name relative to this folder
    <split>/metadata.csv            the same rows, file_name relative to the split folder
                                    (read by datasets.load_dataset("imagefolder", data_dir=...))

Dataset page: https://lila.science/datasets/ohio-small-animals/
Licence: CDLA-Permissive 1.0

Usage:
    python scripts/prepare_datasets/osu_small_animals/download_osu.py --dry-run
    python scripts/prepare_datasets/osu_small_animals/download_osu.py --limit 50
    python scripts/prepare_datasets/osu_small_animals/download_osu.py
"""

import argparse
import csv
import io
import json
import random
import re
import threading
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageOps

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "osu_small_animals"
RAW_DIR = REPO_ROOT / "data" / "raw" / "osu_small_animals"
LABELS_URL = "https://storage.googleapis.com/public-datasets-lila/osu-small-animals/osu-small-animals.json.zip"
IMAGE_BASE_URL = (
    "https://storage.googleapis.com/public-datasets-lila/osu-small-animals/"
)

# Species with >= 500 images at >= 10 sites. Left out: "empty" (only 5 sites, all with
# one camera setup, so the camera would give away the label) and "invertebrate" (a
# catch-all of mostly tiny insects and spiders).
CLASSES = [
    "eastern_gartersnake",
    "song_sparrow",
    "meadow_vole",
    "white-footed_mouse",
    "northern_house_wren",
    "common_five-lined_skink",
    "masked_shrew",
    "eastern_cottontail",
    "long-tailed_weasel",
    "woodland_jumping_mouse",
    "plains_gartersnake",
    "eastern_massasauga",
    "virginia_opossum",
    "common_yellowthroat",
    "n._short-tailed_shrew",
    "dekay's_brownsnake",
]
# Per species and split. Sampling spreads over sequences first, then takes more frames.
CAPS = {"train": 800, "val": 200, "test": 200}
MAX_FRAMES_PER_SEQ = 3
# Share of each species' sequences that should land in each split.
TARGET_SHARE = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEARCH_TRIALS = 20000

MAX_SIDE = 1024
JPEG_QUALITY = 90
MAX_WORKERS = 16
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, doubled after each attempt
TIMEOUT = 60
SEED = 42

METADATA_FIELDS = [
    "image_id",
    "file_name",
    "split",
    "label",
    "site",
    "location",
    "seq_id",
    "frame_num",
    "datetime",
    "source_file",
]

_thread_local = threading.local()


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
    zip_path = raw_dir / "osu-small-animals.json.zip"
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


def site_of(location: str) -> str:
    """Deployment site of a camera, so that neighbouring cameras share a split.

    KPC1/KPC2/KPC3 -> KPC, MLN2A -> MLN, BIWA2E/BIWA4S -> BIWA, CBNP1N -> CBNP,
    OSTN/OSTS -> OST; the 2022 grid keeps its grid point: KILC4N/KILC4S -> KILC4.
    """
    if location.startswith("KIL"):
        return re.match(r"KIL[A-Z]+\d+", location).group()
    if any(c.isdigit() for c in location):
        return re.match(r"[A-Z]+", location).group()
    return location[:-1]  # OSTN -> OST, DORRS -> DORR


def local_name(file_name: str) -> str:
    """Flatten the source path into a unique, shell-friendly file name."""
    stem = Path(file_name.removeprefix("Images/")).with_suffix("").as_posix()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") + ".jpg"


def assign_sites(seqs_per_site: dict[str, Counter]) -> dict[str, str]:
    """Randomly search a site -> split assignment whose per-species shares are closest to the target."""
    rng = random.Random(SEED)
    sites = sorted(seqs_per_site)
    totals = Counter()
    for counts in seqs_per_site.values():
        totals.update(counts)

    def score(assignment: dict[str, str]) -> float:
        per_split = {s: Counter() for s in TARGET_SHARE}
        for site, split in assignment.items():
            per_split[split].update(seqs_per_site[site])
        loss = 0.0
        for species in CLASSES:
            for split, target in TARGET_SHARE.items():
                loss += (per_split[split][species] / totals[species] - target) ** 2
                if split != "train" and per_split[split][species] < 10:
                    loss += 1.0  # every species needs a few sequences in val and test
        return loss

    best, best_loss = None, float("inf")
    splits, weights = list(TARGET_SHARE), list(TARGET_SHARE.values())
    for _ in range(SPLIT_SEARCH_TRIALS):
        assignment = {site: rng.choices(splits, weights)[0] for site in sites}
        loss = score(assignment)
        if loss < best_loss:
            best, best_loss = assignment, loss
    # Then move single sites between splits while that improves the score.
    improved = True
    while improved:
        improved = False
        for site in sites:
            for split in splits:
                candidate = {**best, site: split}
                loss = score(candidate)
                if loss < best_loss - 1e-12:
                    best, best_loss, improved = candidate, loss, True
    print(f"Site split search: loss {best_loss:.4f}")
    return best


def sample_images(images: list[dict], cap: int, rng: random.Random) -> list[dict]:
    """Take up to `cap` images: one frame from every sequence first, then a second, then a third."""
    by_seq = defaultdict(list)
    for im in images:
        by_seq[im["seq_id"]].append(im)
    picks_per_seq = []
    for frames in by_seq.values():
        frames.sort(key=lambda im: im["frame_num"])
        n = min(MAX_FRAMES_PER_SEQ, len(frames))
        # evenly spaced, so a long sequence is not represented by three neighbouring frames
        idx = sorted({round(k * (len(frames) - 1) / max(n - 1, 1)) for k in range(n)})
        picks_per_seq.append([frames[i] for i in idx])
    rng.shuffle(picks_per_seq)

    selected = []
    for k in range(MAX_FRAMES_PER_SEQ):
        for picks in picks_per_seq:
            if k < len(picks) and len(selected) < cap:
                selected.append(picks[k])
    return selected


def build_manifest(raw_dir: Path) -> list[dict]:
    """Return one row per selected image, sorted by output path."""
    labels = load_labels(raw_dir)
    categories = {c["id"]: c["name"] for c in labels["categories"]}
    species_of = {
        a["image_id"]: categories[a["category_id"]] for a in labels["annotations"]
    }
    print(f"Label JSON: {len(labels['images']):,} images, {len(categories)} categories")

    images = [im for im in labels["images"] if species_of[im["id"]] in CLASSES]
    seqs_per_site = defaultdict(Counter)
    for _seq_id, species, site in {
        (im["seq_id"], species_of[im["id"]], site_of(im["location"])) for im in images
    }:
        seqs_per_site[site][species] += 1
    split_of_site = assign_sites(seqs_per_site)

    groups = defaultdict(list)
    for im in images:
        groups[(split_of_site[site_of(im["location"])], species_of[im["id"]])].append(
            im
        )

    rng = random.Random(SEED)
    rows = []
    for (split, species), group in sorted(groups.items()):
        for im in sample_images(group, CAPS[split], rng):
            name = local_name(im["file_name"])
            rows.append(
                {
                    "image_id": name.removesuffix(".jpg"),
                    "file_name": f"{split}/{species}/{name}",
                    "split": split,
                    "label": species,
                    "site": site_of(im["location"]),
                    "location": im["location"],
                    "seq_id": im["seq_id"],
                    "frame_num": im["frame_num"],
                    "datetime": im["datetime"],
                    "source_file": im["file_name"],
                }
            )
    rows.sort(key=lambda r: r["file_name"])
    assert len({r["file_name"] for r in rows}) == len(rows), (
        "output file names are not unique"
    )

    print_split_table(rows, "Selected images")
    for split in CAPS:
        split_sites = sorted(s for s, sp in split_of_site.items() if sp == split)
        print(f"  {split} sites ({len(split_sites)}): {', '.join(split_sites)}")
    return rows


def print_split_table(rows: list[dict], title: str) -> None:
    images = Counter((r["split"], r["label"]) for r in rows)
    seqs, sites = defaultdict(set), defaultdict(set)
    for r in rows:
        seqs[(r["split"], r["label"])].add(r["seq_id"])
        sites[(r["split"], r["label"])].add(r["site"])
    print(f"\n{title}: images (sequences, sites)")
    print(f"  {'species':<26}" + "".join(f"{s:>18}" for s in CAPS))
    for species in CLASSES:
        cells = [
            f"{images[(s, species)]} ({len(seqs[(s, species)])}, {len(sites[(s, species)])})"
            for s in CAPS
        ]
        print(f"  {species:<26}" + "".join(f"{c:>18}" for c in cells))
    print(
        f"  {'total':<26}"
        + "".join(f"{sum(images[(s, c)] for c in CLASSES):>18}" for s in CAPS)
    )


# ---------------------------------------------------------------------------
# Step 2: download and downscale
# ---------------------------------------------------------------------------
def select_limited(rows: list[dict], limit: int) -> list[dict]:
    """Pick `limit` rows round-robin over (split, label) so a test run covers all of them."""
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["label"])].append(row)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    ordered = [groups[k] for k in sorted(groups)]
    while len(selected) < limit and any(ordered):
        for group in ordered:
            if group and len(selected) < limit:
                selected.append(group.pop())
    return selected


def download_one(row: dict, out_dir: Path, max_side: int) -> str | None:
    """Fetch, downscale and save one image. Returns an error message or None."""
    dest_path = out_dir / row["file_name"]
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    url = IMAGE_BASE_URL + quote(row["source_file"], safe="/")

    error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = get_session().get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
            img.draft(
                "RGB", (max_side, max_side)
            )  # decode at reduced size, never below max_side
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail(
                (max_side, max_side), Image.Resampling.LANCZOS
            )  # never upscales
            img.save(tmp_path, format="JPEG", quality=JPEG_QUALITY)
            tmp_path.replace(dest_path)
            return None
        except Exception as e:  # any failure is retried, then recorded
            error = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * 2**attempt)
    tmp_path.unlink(missing_ok=True)
    return error


def download_all(
    rows: list[dict], out_dir: Path, raw_dir: Path, workers: int, max_side: int
) -> None:
    todo = [r for r in rows if not (out_dir / r["file_name"]).exists()]
    print(
        f"\nDownload: {len(rows):,} selected, {len(rows) - len(todo):,} already on disk, {len(todo):,} to fetch"
    )

    failed = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, r, out_dir, max_side): r for r in todo}
        for i, future in enumerate(as_completed(futures), 1):
            error = future.result()
            if error:
                failed.append(
                    {"source_file": futures[future]["source_file"], "error": error}
                )
            if i % 500 == 0 or i == len(todo):
                rate = i / (time.time() - start)
                print(
                    f"  {i:,}/{len(todo):,}  failed={len(failed)}  {rate:.1f} img/s",
                    flush=True,
                )

    with (raw_dir / "failed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_file", "error"])
        writer.writeheader()
        writer.writerows(failed)
    print(
        f"Fetched {len(todo) - len(failed):,}, failed {len(failed):,} (see {raw_dir / 'failed.csv'})"
    )


def remove_stale(rows: list[dict], out_dir: Path) -> None:
    """Delete images and folders on disk that are not in the manifest (e.g. from an earlier selection)."""
    wanted = {out_dir / r["file_name"] for r in rows}
    stale = [
        p for split in CAPS for p in (out_dir / split).rglob("*.jpg") if p not in wanted
    ]
    for path in stale:
        path.unlink()
    for split in CAPS:
        for folder in (out_dir / split).glob("*/"):
            if not any(folder.iterdir()):
                folder.rmdir()
    if stale:
        print(f"Removed {len(stale):,} images that are no longer in the manifest")


# ---------------------------------------------------------------------------
# Step 3: verify and write metadata
# ---------------------------------------------------------------------------
def verify(rows: list[dict], out_dir: Path, raw_dir: Path, max_side: int) -> None:
    """Check every saved file, then write the metadata files for the images on disk only."""
    unreadable, too_large, present, total_bytes = [], [], [], 0
    for row in rows:
        path = out_dir / row["file_name"]
        if not path.exists():
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

    # Metadata lists only images that exist, so the ImageFolder loader never sees a missing file.
    # `datasets` reads the per-split files; the root file is the combined view.
    write_csv(out_dir / "metadata.csv", present)
    for split in CAPS:
        split_rows = [
            {**r, "file_name": r["file_name"].removeprefix(f"{split}/")}
            for r in present
            if r["split"] == split
        ]
        write_csv(out_dir / split / "metadata.csv", split_rows)

    print(
        f"\nVerify: {len(present):,} readable, {len(rows) - len(present) - len(unreadable):,} not on disk"
    )
    print(
        f"Unreadable: {len(unreadable)}; longest side > {max_side}px: {len(too_large)}"
    )
    print(f"Size on disk: {total_bytes / 1e9:.2f} GB")
    for name in unreadable[:20]:
        print(f"  unreadable: {name}")
    print_split_table(present, "On disk")

    make_contact_sheet(present, out_dir, raw_dir)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".part")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*METADATA_FIELDS, "width", "height"])
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def make_contact_sheet(
    rows: list[dict], out_dir: Path, raw_dir: Path, per_class: int = 6, cell: int = 320
) -> None:
    """Save a grid with random training images per species, one row each."""
    rng = random.Random(SEED)
    cell_h, header = cell * 3 // 4, 20
    sheet = Image.new(
        "RGB", (cell * per_class, (cell_h + header) * len(CLASSES)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for i, species in enumerate(CLASSES):
        candidates = [
            r for r in rows if r["label"] == species and r["split"] == "train"
        ]
        top = i * (cell_h + header)
        draw.text((6, top + 4), f"{species}  (train n={len(candidates)})", fill="black")
        for j, row in enumerate(
            rng.sample(candidates, min(per_class, len(candidates)))
        ):
            with Image.open(out_dir / row["file_name"]) as img:
                thumb = ImageOps.pad(img.convert("RGB"), (cell, cell_h), color="black")
            sheet.paste(thumb, (j * cell, top + header))
            draw.text((j * cell + 4, top + header + 4), row["location"], fill="yellow")
    path = raw_dir / "contact_sheet.jpg"
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
        help="output directory (default: data/osu_small_animals)",
    )
    parser.add_argument(
        "--raw", type=Path, default=RAW_DIR, help="labels, failures, contact sheet"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the selection and stop"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="download only N images, spread over splits/labels",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-side", type=int, default=MAX_SIDE)
    args = parser.parse_args()

    rows = build_manifest(args.raw)
    if args.dry_run:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    remove_stale(rows, args.out)
    selected = select_limited(rows, args.limit) if args.limit else rows
    download_all(selected, args.out, args.raw, args.workers, args.max_side)
    verify(rows, args.out, args.raw, args.max_side)


if __name__ == "__main__":
    main()
