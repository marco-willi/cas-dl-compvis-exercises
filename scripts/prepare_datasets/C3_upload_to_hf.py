"""Step C3 — Upload curated datasets to Hugging Face Hub.

Requires:
    huggingface-cli login   (or HF_TOKEN env var set)

Usage:
    python scripts/prepare_datasets/C3_upload_to_hf.py kgalagadi
    python scripts/prepare_datasets/C3_upload_to_hf.py cct20
    python scripts/prepare_datasets/C3_upload_to_hf.py ser_balanced
    python scripts/prepare_datasets/C3_upload_to_hf.py abo_furniture
    python scripts/prepare_datasets/C3_upload_to_hf.py osu_small_animals
    python scripts/prepare_datasets/C3_upload_to_hf.py all
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

DATASETS = {
    "kgalagadi": {
        "repo_id": "marco-willi/camera-trap-kgalagadi",
        "archive": DATA_DIR / "kgalagadi.tar.gz",
        "checksum": DATA_DIR / "kgalagadi.tar.gz.sha256",
        "card": Path(__file__).resolve().parent / "kgalagadi_card.md",
    },
    "cct20": {
        "repo_id": "marco-willi/camera-trap-cct20",
        "archive": DATA_DIR / "cct20.tar.gz",
        "checksum": DATA_DIR / "cct20.tar.gz.sha256",
        "card": Path(__file__).resolve().parent / "cct20_card.md",
    },
    "ser_balanced": {
        "repo_id": "marco-willi/ser_balanced",
        "archive": DATA_DIR / "ser_balanced.tar.gz",
        "checksum": DATA_DIR / "ser_balanced.tar.gz.sha256",
        "card": Path(__file__).resolve().parent / "ser_balanced_card.md",
    },
    "abo_furniture": {
        "repo_id": "marco-willi/abo_furniture",
        "archive": DATA_DIR / "abo_furniture.tar.gz",
        "checksum": DATA_DIR / "abo_furniture.tar.gz.sha256",
        "card": Path(__file__).resolve().parent / "abo_furniture_card.md",
    },
    "ser_sampled": {
        "repo_id": "marco-willi/ser_sampled",
        "archive": DATA_DIR / "ser_sampled.tar.gz",
        "checksum": DATA_DIR / "ser_sampled.tar.gz.sha256",
        "card": Path(__file__).resolve().parent / "ser_sampled_card.md",
    },
    "ser_sampled_cropped": {
        "repo_id": "marco-willi/ser_sampled_cropped",
        "archive": DATA_DIR / "ser_sampled_cropped.tar.gz",
        "checksum": DATA_DIR / "ser_sampled_cropped.tar.gz.sha256",
        "card": Path(__file__).resolve().parent / "ser_sampled_card.md",
    },
    "osu_small_animals": {
        "repo_id": "marco-willi/camera-trap-osu-small-animals",
        "archive": DATA_DIR / "osu_small_animals.tar.gz",
        "checksum": DATA_DIR / "osu_small_animals.tar.gz.sha256",
        "card": Path(__file__).resolve().parent / "osu_small_animals_card.md",
    },
}


def upload(dataset: str):
    cfg = DATASETS[dataset]
    repo_id = cfg["repo_id"]
    archive = cfg["archive"]
    checksum_file = cfg["checksum"]
    card_file = cfg["card"]

    assert archive.exists(), f"Archive not found: {archive}"

    api = HfApi()

    # Create repo if it doesn't exist
    print(f"Creating/verifying repo: {repo_id}")
    create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)

    # Upload archive
    size_mb = archive.stat().st_size / 1e6
    print(f"Uploading {archive.name} ({size_mb:.0f} MB) ...")
    api.upload_file(
        path_or_fileobj=str(archive),
        path_in_repo=archive.name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  Done.")

    # Upload checksum
    print(f"Uploading {checksum_file.name} ...")
    api.upload_file(
        path_or_fileobj=str(checksum_file),
        path_in_repo=checksum_file.name,
        repo_id=repo_id,
        repo_type="dataset",
    )

    # Upload dataset card as README.md
    print("Uploading README.md (dataset card) ...")
    api.upload_file(
        path_or_fileobj=str(card_file),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    # Tag at v1.0
    print("Tagging at v1.0 ...")
    try:
        api.create_tag(repo_id, tag="v1.0", repo_type="dataset")
    except Exception as e:
        print(f"  Tag already exists or failed: {e}")

    print(f"\n✓ Uploaded: https://huggingface.co/datasets/{repo_id}")

    # Verify unauthenticated download
    print("\nVerifying unauthenticated download ...")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, archive.name, repo_type="dataset")
    print(f"  Downloaded to: {path}")
    print("  ✓ Unauthenticated download works.")


if __name__ == "__main__":
    targets = sys.argv[1:] if sys.argv[1:] else ["all"]
    if targets == ["all"]:
        targets = list(DATASETS.keys())

    for ds in targets:
        assert ds in DATASETS, (
            f"Unknown dataset '{ds}'. Choose from: {list(DATASETS.keys())}"
        )
        print(f"\n{'=' * 60}")
        print(f"Uploading: {ds}")
        print(f"{'=' * 60}")
        upload(ds)

    print("\nAll done.")
