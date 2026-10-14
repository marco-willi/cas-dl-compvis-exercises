"""Download a dataset archive, trying Google Drive first and Hugging Face as a fallback.

Sources are looked up in ``scripts/dataset_registry.py``. Neither path requires
authentication — the Drive files are publicly shared and the HF repos are public.

CLI usage:
    python scripts/download_dataset.py ser_balanced.tar.gz
    python scripts/download_dataset.py ser_balanced.tar.gz --dest data/
    python scripts/download_dataset.py ser_balanced.tar.gz --source hf

Programmatic usage:
    from scripts.download_dataset import fetch
    path = fetch("ser_balanced.tar.gz", dest="data/")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dataset_registry import GDRIVE_IDS, HF_REPOS  # noqa: E402


def _fetch_gdrive(name: str, dest: Path) -> Path:
    import gdown

    file_id = GDRIVE_IDS[name]
    out = dest / name
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(out), quiet=False, fuzzy=True)
    assert out.exists(), f"gdown reported success but {out} is missing"
    return out


def _fetch_hf(name: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    repo_id = HF_REPOS[name]
    cached = hf_hub_download(repo_id, name, repo_type="dataset")
    out = dest / name
    if out.resolve() != Path(cached).resolve():
        out.write_bytes(Path(cached).read_bytes())
    return out


def fetch(name: str, dest: str | Path = "data", source: str = "auto") -> Path:
    """Download ``name`` into ``dest``. Returns path to the local archive.

    ``source`` is one of ``"auto"`` (Drive then HF), ``"gdrive"``, or ``"hf"``.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if source not in {"auto", "gdrive", "hf"}:
        raise ValueError(f"Unknown source: {source!r}")

    if source in {"auto", "gdrive"} and name in GDRIVE_IDS:
        try:
            return _fetch_gdrive(name, dest)
        except Exception as exc:
            if source == "gdrive":
                raise
            print(f"[download] Drive failed ({exc}); falling back to HF Hub.")

    if name not in HF_REPOS:
        raise KeyError(f"No HF fallback registered for {name!r}")
    return _fetch_hf(name, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="archive filename, e.g. ser_balanced.tar.gz")
    parser.add_argument("--dest", default="data", help="download directory")
    parser.add_argument(
        "--source",
        choices=["auto", "gdrive", "hf"],
        default="auto",
        help="force a specific source",
    )
    args = parser.parse_args()
    path = fetch(args.name, dest=args.dest, source=args.source)
    print(f"\n✓ Saved to: {path}")


if __name__ == "__main__":
    main()
