"""Step 6 — Notebook download helper (reference copy).

This file is NOT imported by anything. It contains the self-contained
load_deepfashion() function to be inlined into exercise notebooks
(Exercises 3 and 5, when the DeepFashion variant is used instead of ABO).

Copy the function body into the notebook setup cell.

IMPORTANT: The Google Drive file ID must be filled in by the instructor
after uploading deepfashion_classroom_v1_internal.zip to a restricted
Google Drive folder.

SHA-256: a9e8327ee0980f08b4fbc739b75a0a5ba0845e16cbb70c8e23959fd2b96f8fa9
"""

# ============================================================
# INLINE THIS FUNCTION IN EXERCISE NOTEBOOKS
# ============================================================

from pathlib import Path


def load_deepfashion(
    data_path: Path,
    gdrive_file_id: str | None = None,
) -> Path:
    """Load the DeepFashion classroom subset (internal, restricted distribution).

    If data_path/deepfashion/ already exists, returns immediately (idempotent).
    Otherwise downloads the ZIP from a restricted Google Drive link using gdown.

    Args:
        data_path:      Local directory where the dataset will be extracted.
        gdrive_file_id: Google Drive file ID of the internal ZIP archive.
                        Provided by the course instructor.
                        Example: "1aBcDeFgHiJkLmNoPqRsTuVwXyZ012345"

    Returns:
        Path to the extracted dataset directory (ImageFolder layout).
        Use with torchvision.datasets.ImageFolder(path / "train").

    Raises:
        FileNotFoundError: If the dataset is not found locally and no
                           gdrive_file_id is supplied.
        ImportError:       If gdown is not installed. Run: pip install gdown
    """
    dataset_dir = data_path / "deepfashion"

    if dataset_dir.exists():
        print(f"Dataset already present: {dataset_dir}")
        return dataset_dir

    if gdrive_file_id is None:
        raise FileNotFoundError(
            "DeepFashion classroom dataset not found locally.\n"
            "Ask your instructor for the Google Drive file ID and pass it as\n"
            "  load_deepfashion(data_path, gdrive_file_id='<ID>')"
        )

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download the DeepFashion dataset.\n"
            "Install it with:  pip install gdown"
        ) from exc

    import zipfile

    zip_path = data_path / "deepfashion_classroom_v1_internal.zip"
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"

    print("Downloading DeepFashion classroom subset from Google Drive ...")
    gdown.download(url, str(zip_path), quiet=False)

    print(f"Extracting to {data_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_path)

    # Clean up archive after extraction to save disk space
    zip_path.unlink(missing_ok=True)

    assert dataset_dir.exists(), f"Extraction failed — {dataset_dir} not found"
    print(f"Ready: {dataset_dir}")
    return dataset_dir
