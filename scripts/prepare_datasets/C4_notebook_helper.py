"""Step C4 — Notebook download helper (reference copy).

This file is NOT imported by the notebooks — they are self-contained for Colab.
The ``ensure_dataset`` function below is the canonical copy that gets inlined
into each exercise notebook's setup cell.

Keep this file in sync with the inlined copies. IDs live in
``scripts/dataset_registry.py``; notebooks inline only the specific IDs they
need, not the whole registry.
"""

from pathlib import Path


def ensure_dataset(
    data_path: Path,
    archive_name: str,
    extracted_subdir: str,
    *,
    gdrive_id: str | None = None,
    hf_repo: str | None = None,
) -> Path:
    """Check-then-download a dataset archive. Returns the extracted directory.

    Tries Google Drive first (via ``gdown``), falls back to Hugging Face Hub
    (via ``huggingface_hub.hf_hub_download``). Supports ``.tar.gz`` and ``.zip``
    archives. Idempotent — if ``data_path / extracted_subdir`` already exists,
    returns immediately without re-downloading.

    Args:
        data_path: Local directory where the archive is placed and extracted.
        archive_name: Filename of the archive (e.g. ``"ser_balanced.tar.gz"``).
        extracted_subdir: Path (relative to ``data_path``) of the directory that
            the archive creates on extraction.
        gdrive_id: Google Drive file ID. If ``None``, Drive is skipped.
        hf_repo: Hugging Face dataset repo (e.g. ``"marco-willi/ser_balanced"``).
            If ``None``, HF fallback is skipped.

    Returns:
        ``data_path / extracted_subdir``.

    Raises:
        RuntimeError: If the dataset is missing and neither source is available
            or both sources fail.
    """
    data_path = Path(data_path)
    dataset_dir = data_path / extracted_subdir
    if dataset_dir.exists():
        print(f"Dataset already present: {dataset_dir}")
        return dataset_dir

    data_path.mkdir(parents=True, exist_ok=True)
    archive_path = data_path / archive_name

    if not archive_path.exists():
        errors = []
        if gdrive_id is not None:
            try:
                import gdown

                url = f"https://drive.google.com/uc?id={gdrive_id}"
                print(f"Downloading {archive_name} from Google Drive ...")
                gdown.download(url, str(archive_path), quiet=False, fuzzy=True)
                if not archive_path.exists():
                    raise RuntimeError("gdown reported success but file is missing")
            except Exception as exc:
                errors.append(f"gdown: {exc}")
                archive_path.unlink(missing_ok=True)

        if not archive_path.exists() and hf_repo is not None:
            try:
                from huggingface_hub import hf_hub_download

                print(f"Downloading {archive_name} from Hugging Face ({hf_repo}) ...")
                cached = hf_hub_download(hf_repo, archive_name, repo_type="dataset")
                if Path(cached).resolve() != archive_path.resolve():
                    archive_path.write_bytes(Path(cached).read_bytes())
            except Exception as exc:
                errors.append(f"hf_hub_download: {exc}")

        if not archive_path.exists():
            sources = ", ".join(errors) if errors else "no sources configured"
            raise RuntimeError(
                f"Could not fetch {archive_name}. Tried: {sources}. "
                "Check scripts/dataset_registry.py for valid IDs."
            )

    print(f"Extracting {archive_path.name} to {data_path} ...")
    if archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
        import tarfile

        with tarfile.open(archive_path) as tar:
            tar.extractall(data_path)
    elif archive_name.endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(data_path)
    else:
        raise ValueError(f"Unsupported archive format: {archive_name}")

    assert dataset_dir.exists(), (
        f"Extraction of {archive_path.name} did not produce {dataset_dir}. "
        "Check that extracted_subdir matches the archive's top-level directory."
    )
    print(f"Ready: {dataset_dir}")
    return dataset_dir


# ---------------------------------------------------------------------------
# Available datasets — inline snippets for exercise notebooks
#
# Notebooks are self-contained (no imports from this file). Copy the relevant
# snippet into the notebook's setup cell, replacing the GDrive ID placeholder
# with the value from scripts/dataset_registry.py once the archive is uploaded.
#
# Default camera-trap / retrieval datasets (GDrive + HF fallback):
# ---------------------------------------------------------------------------
#
#   ser_balanced (default for 01, 02, 04, 04_lora, 05):
#     dataset_dir = ensure_dataset(
#         DATA_PATH, "ser_balanced.tar.gz", "ser_balanced",
#         gdrive_id="1iRlZue4-Udg_lA9RCYtg3zhctqbbPhX7",
#         hf_repo="marco-willi/ser_balanced",
#     )
#
#   cct20 (alternative for 01, 02, 04, 04_lora, 05):
#     dataset_dir = ensure_dataset(
#         DATA_PATH, "cct20.tar.gz", "cct20",
#         gdrive_id="105DkEQcFhgWsQEzKh6p-u2QMMuUc2yt2",
#         hf_repo="marco-willi/camera-trap-cct20",
#     )
#
#   kgalagadi (alternative for 01, 02, 04, 04_lora, 05 — heavy class imbalance):
#     dataset_dir = ensure_dataset(
#         DATA_PATH, "kgalagadi.tar.gz", "kgalagadi",
#         gdrive_id="129vX_GF4vUgwRlyLpx5BNPf6lI2n89Wp",
#     )
#
#   abo_furniture (default for 03_retrieval):
#     dataset_dir = ensure_dataset(
#         DATA_PATH, "abo_furniture.tar.gz", "abo",
#         gdrive_id="1ClnTzR1plXXzKQslsK4YaQzzxCHU44cK",
#         hf_repo="marco-willi/abo_furniture",
#     )
#
# ---------------------------------------------------------------------------
# Optional alternative datasets (GDrive only — no HF mirror due to licensing)
# Suitable for exercises 01, 02, 04, 04_lora, 05. NOT for 03_retrieval.
# ---------------------------------------------------------------------------
#
#   cats_vs_dogs (~25k images, 2 classes: Cat / Dog):
#     dataset_dir = ensure_dataset(
#         DATA_PATH, "cats_vs_dogs.tar.gz", "cats_vs_dogs",
#         gdrive_id="<cats_vs_dogs-drive-id>",   # fill from dataset_registry.py
#     )
#     # Images are variable size — apply transforms.Resize(256) + CenterCrop(224)
#
#   concrete_cracks (~40k images, 2 classes: Negative / Positive, 227x227 px):
#     dataset_dir = ensure_dataset(
#         DATA_PATH, "concrete_cracks.tar.gz", "concrete_cracks",
#         gdrive_id="<concrete_cracks-drive-id>",  # fill from dataset_registry.py
#     )
#     # Images are 227x227 — transforms.Resize(224) or CenterCrop(224) suffices
#
#   eurosat (~27k images, 10 land-use classes, 64x64 px satellite RGB):
#     dataset_dir = ensure_dataset(
#         DATA_PATH, "eurosat.tar.gz", "eurosat",
#         gdrive_id="<eurosat-drive-id>",          # fill from dataset_registry.py
#     )
#     # IMPORTANT: images are 64x64 px — always apply transforms.Resize(224)
#     #            (or Resize(256) + CenterCrop(224)) for ImageNet-pretrained models
# ---------------------------------------------------------------------------
