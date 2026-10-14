#!/usr/bin/env python3
"""Build the student-facing `main` branch from the current `dev` branch.

Reads ``publish.yml`` to decide which notebooks and files to include,
generates a clean ``README.md`` with Colab badges, and force-updates
the ``main`` branch.

Usage (local):
    python scripts/publish_to_main.py          # dry-run: prints what would be published
    python scripts/publish_to_main.py --push   # actually update the main branch

Usage (CI):
    Called by .github/workflows/publish.yml with --push.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "publish.yml"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from strip_solutions import process_notebook  # noqa: E402

# ── Helpers ──────────────────────────────────────────────────────────────────


def run(
    cmd: list[str],
    *,
    display_cmd: list[str] | None = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Run a subprocess, printing the command and raising on failure."""
    shown = display_cmd or cmd
    print(f"  $ {' '.join(shown)}")
    return subprocess.run(cmd, check=True, **kwargs)


def generate_requirements(staging: Path) -> None:
    """Write a pip-compatible requirements.txt from [project].dependencies.

    Emits only direct dependencies (not the full transitive closure) so the file
    stays short and human-readable. Students who want an exact pinned env can use
    uv.lock, which is also published to main.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    lines = list(pyproject["project"]["dependencies"])

    out = staging / "requirements.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"  Generated requirements.txt ({len(lines)} direct deps)")


def get_git_config_value(key: str) -> str | None:
    """Return a git config value from the current environment, if available."""
    result = subprocess.run(
        ["git", "config", "--get", key],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def configure_commit_identity(repo_path: Path) -> None:
    """Ensure git commits in the temporary publish repo have an identity."""
    actor = os.environ.get("GITHUB_ACTOR")
    name = (
        os.environ.get("GIT_AUTHOR_NAME")
        or os.environ.get("GIT_COMMITTER_NAME")
        or get_git_config_value("user.name")
        or actor
        or "github-actions[bot]"
    )
    email = (
        os.environ.get("GIT_AUTHOR_EMAIL")
        or os.environ.get("GIT_COMMITTER_EMAIL")
        or get_git_config_value("user.email")
        or (
            f"{actor}@users.noreply.github.com"
            if actor and actor != "github-actions[bot]"
            else "github-actions[bot]@users.noreply.github.com"
        )
    )

    run(["git", "-C", str(repo_path), "config", "user.name", name])
    run(["git", "-C", str(repo_path), "config", "user.email", email])


def get_authenticated_remote_url(remote_url: str) -> str:
    """Return an HTTPS remote URL augmented with a GitHub token when available."""
    token = (
        os.environ.get("PUBLISH_GITHUB_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
    )
    prefix = "https://github.com/"

    if not token or not remote_url.startswith(prefix):
        return remote_url

    return (
        f"https://x-access-token:{token}@github.com/{remote_url.removeprefix(prefix)}"
    )


def find_exercise_notebook(notebook_dir: Path) -> str | None:
    """Return the exercise notebook name in a directory, preferring exercise.ipynb."""
    exercise = notebook_dir / "exercise.ipynb"
    if exercise.exists():
        return exercise.name
    # Fallback: first .ipynb that isn't a solution or checkpoint
    for f in sorted(notebook_dir.glob("*.ipynb")):
        if ".ipynb_checkpoints" not in str(f) and f.name != "solution.ipynb":
            return f.name
    return None


def get_manifest_list(config: dict, key: str) -> list:
    """Return a manifest list field, normalizing null to an empty list."""
    value = config.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        print(f"ERROR: publish.yml field '{key}' must be a list")
        sys.exit(1)
    return value


# ── README generation ────────────────────────────────────────────────────────


def generate_readme(config: dict) -> str:
    """Build a student-facing README.md from the manifest."""
    repo = config["github_repo"]
    notebooks = get_manifest_list(config, "exercises")

    lines = [
        "# Exercises for CAS Deep Learning - Module Computer Vision with Deep Learning (Part 1)",
        "",
        "This repository contains exercises for the CAS Deep Learning - Module Computer Vision with Deep Learning (Part 1).",
        "",
        "There are several ways to work on the assignments:",
        "",
        "- Google Colab (easiest)",
        "- Local install with pip",
        "- Local install with uv (reproducible env from the lockfile)",
        "",
        "## Google Colab",
        "",
        "Use Google Colab by clicking on the links below.",
        "",
    ]

    badge = "https://colab.research.google.com/assets/colab-badge.svg"

    def colab_url(path: str) -> str:
        return f"https://colab.research.google.com/github/{repo}/blob/main/{path}"

    for nb in notebooks:
        folder = nb["folder"]
        title = nb["title"]
        nb_dir = REPO_ROOT / "exercises" / folder
        nb_file = find_exercise_notebook(nb_dir)
        if nb_file is None:
            print(f"  WARNING: no .ipynb found in exercises/{folder}, skipping")
            continue

        # Extract exercise number from folder name (e.g. "01" from "01_pytorch_and_images")
        number = folder.split("_")[0]

        exercise_url = colab_url(f"exercises/{folder}/{nb_file}")
        solution_path = nb_dir / "solution.ipynb"
        solution_url = (
            colab_url(f"exercises/{folder}/solution.ipynb")
            if solution_path.exists()
            else None
        )

        lines.extend(
            [
                f"### Exercise {number} - {title}",
                "",
                "Open in Google Colab:",
                "",
                f"- Exercise: [![Open In Colab]({badge})]({exercise_url})",
            ]
        )
        if solution_url is not None:
            lines.append(f"    - Solution: [![Open In Colab]({badge})]({solution_url})")
        lines.append("")

    lines.extend(
        [
            "## Local Install",
            "",
            "### Option A — pip",
            "",
            "Install the direct dependencies into your current Python environment:",
            "",
            "```",
            "pip install -r requirements.txt",
            "```",
            "",
            "### Option B — uv",
            "",
            "uv installs the exact pinned versions from `uv.lock`, giving a"
            " reproducible environment. If you do not have uv yet, follow the"
            " official install instructions: <https://docs.astral.sh/uv/>.",
            "",
            "```",
            "uv sync --no-dev",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


# ── Build main branch content ────────────────────────────────────────────────


def build_main(config: dict, staging: Path) -> None:
    """Copy published files into a staging directory."""
    notebooks = get_manifest_list(config, "exercises")
    includes = get_manifest_list(config, "include")

    # 1. Generate exercise notebooks from solution notebooks
    for nb in notebooks:
        folder = nb["folder"]
        solution = REPO_ROOT / "exercises" / folder / "solution.ipynb"
        if solution.exists():
            process_notebook(solution)

    # 2. Generate README.md (after exercise notebooks exist for Colab link lookup)
    readme = generate_readme(config)
    (staging / "README.md").write_text(readme)
    print(f"  Generated README.md ({len(notebooks)} exercises)")

    # 2b. Export dependencies to requirements.txt for pip-based installs on main
    generate_requirements(staging)

    # 3. Copy exercise directories (includes both solution.ipynb and exercise.ipynb)
    nb_dest = staging / "exercises"
    nb_dest.mkdir()
    for nb in notebooks:
        folder = nb["folder"]
        src = REPO_ROOT / "exercises" / folder
        if not src.is_dir():
            print(f"  WARNING: exercises/{folder} does not exist, skipping")
            continue
        dst = nb_dest / folder
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".ipynb_checkpoints"))
        print(f"  Copied exercises/{folder}/")

    # 4. Copy extra includes
    for item in includes:
        src = REPO_ROOT / item
        dst = staging / item
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
            )
            print(f"  Copied {item} (directory)")
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  Copied {item}")
        else:
            print(f"  WARNING: {item} does not exist, skipping")


# ── Git operations ───────────────────────────────────────────────────────────


def push_to_main(staging: Path, dry_run: bool) -> None:
    """Commit the staging directory contents to the `main` branch and push."""
    if dry_run:
        print("\nDry run — files that would be on main:")
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(staging)}")
        return

    # Get the remote URL from the current repo
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print("ERROR: no git remote 'origin' found")
        sys.exit(1)
    remote_url = result.stdout.strip()
    auth_remote_url = get_authenticated_remote_url(remote_url)

    # Work in a temporary clone to avoid touching the dev worktree
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "repo"

        # Check if main branch exists on the remote
        check = subprocess.run(
            ["git", "ls-remote", "--heads", auth_remote_url, "main"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        main_exists = bool(check.stdout.strip())

        if main_exists:
            # Clone only the main branch (shallow)
            run(
                [
                    "git",
                    "clone",
                    "--branch",
                    "main",
                    "--single-branch",
                    "--depth",
                    "1",
                    auth_remote_url,
                    str(tmp_path),
                ],
                display_cmd=[
                    "git",
                    "clone",
                    "--branch",
                    "main",
                    "--single-branch",
                    "--depth",
                    "1",
                    remote_url,
                    str(tmp_path),
                ],
            )
        else:
            # Create a fresh repo with an orphan main branch
            run(["git", "init", str(tmp_path)])
            run(["git", "-C", str(tmp_path), "checkout", "--orphan", "main"])
            run(
                [
                    "git",
                    "-C",
                    str(tmp_path),
                    "remote",
                    "add",
                    "origin",
                    auth_remote_url,
                ],
                display_cmd=[
                    "git",
                    "-C",
                    str(tmp_path),
                    "remote",
                    "add",
                    "origin",
                    remote_url,
                ],
            )

        # Clear existing content (except .git)
        for child in tmp_path.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

        # Copy staged content in
        for child in staging.iterdir():
            dst = tmp_path / child.name
            if child.is_dir():
                shutil.copytree(child, dst)
            else:
                shutil.copy2(child, dst)

        configure_commit_identity(tmp_path)

        # Commit and push
        run(["git", "-C", str(tmp_path), "add", "-A"])

        # Check if there are changes to commit
        diff_result = subprocess.run(
            ["git", "-C", str(tmp_path), "diff", "--cached", "--quiet"]
        )
        if diff_result.returncode == 0:
            print("\nNo changes to publish — main is already up to date.")
            return

        run(
            [
                "git",
                "-C",
                str(tmp_path),
                "commit",
                "-m",
                "Publish exercises from dev branch",
            ]
        )
        run(["git", "-C", str(tmp_path), "push", "origin", "main"])
        print("\nPublished to main.")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Publish dev → main")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Actually push to main (default is dry-run)",
    )
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} not found")
        sys.exit(1)

    config = yaml.safe_load(MANIFEST.read_text())
    if config is None:
        print(f"ERROR: {MANIFEST} is empty")
        sys.exit(1)
    if not isinstance(config, dict):
        print(f"ERROR: {MANIFEST} must contain a YAML mapping at the top level")
        sys.exit(1)
    if "github_repo" not in config:
        print("ERROR: publish.yml is missing required field 'github_repo'")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as staging_dir:
        staging = Path(staging_dir)
        print("Building main branch content...")
        build_main(config, staging)
        push_to_main(staging, dry_run=not args.push)


if __name__ == "__main__":
    main()
