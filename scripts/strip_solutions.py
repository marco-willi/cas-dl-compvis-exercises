#!/usr/bin/env python3
"""Generate exercise notebooks from solution notebooks.

Any code cell tagged with ``solution`` is replaced with a stub so the
solution notebook can remain the single source of truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

STUB_SOURCE = ["# YOUR CODE HERE\n", "raise NotImplementedError\n"]


def load_notebook(path: Path) -> dict:
    """Load a notebook from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_notebook(path: Path, notebook: dict) -> None:
    """Write a notebook with stable formatting."""
    path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def is_solution_cell(cell: dict) -> bool:
    """Return True when a cell is a solution code cell."""
    metadata = cell.get("metadata", {})
    tags = metadata.get("tags", [])
    return cell.get("cell_type") == "code" and "solution" in tags


def strip_solution_cells(notebook: dict) -> tuple[dict, int]:
    """Replace solution cells with student stubs."""
    stripped = deepcopy(notebook)
    replaced = 0

    for cell in stripped.get("cells", []):
        if not is_solution_cell(cell):
            continue

        cell["source"] = STUB_SOURCE.copy()
        cell["execution_count"] = None
        cell["outputs"] = []
        tags = cell.get("metadata", {}).get("tags", [])
        cell["metadata"]["tags"] = [t for t in tags if t != "solution"]
        replaced += 1

    return stripped, replaced


def default_output_path(solution_path: Path) -> Path:
    """Return the default exercise notebook path for a solution notebook."""
    return solution_path.with_name("exercise.ipynb")


def process_notebook(solution_path: Path, output_path: Path | None = None) -> Path:
    """Generate one exercise notebook from one solution notebook."""
    if not solution_path.exists():
        print(f"ERROR: notebook not found: {solution_path}")
        sys.exit(1)

    notebook = load_notebook(solution_path)
    stripped, replaced = strip_solution_cells(notebook)

    destination = output_path or default_output_path(solution_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_notebook(destination, stripped)
    print(
        f"Generated {destination.relative_to(destination.parent.parent.parent)}"
        f" from {solution_path.relative_to(solution_path.parent.parent.parent)}"
        f" ({replaced} solution cells replaced)"
    )
    return destination


def iter_solution_notebooks(root: Path) -> list[Path]:
    """Return all solution notebooks under the notebooks directory."""
    return sorted((root / "exercises").glob("**/solution.ipynb"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exercise notebooks")
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Path(s) to solution.ipynb files to strip",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate exercise notebooks for all exercises/**/solution.ipynb",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used with --all (default: current working directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all and args.notebooks:
        print("ERROR: pass notebook paths or --all, not both")
        sys.exit(1)
    if not args.all and not args.notebooks:
        print("ERROR: pass at least one solution notebook path or use --all")
        sys.exit(1)

    if args.all:
        notebooks = iter_solution_notebooks(args.root)
        if not notebooks:
            print("ERROR: no solution notebooks found")
            sys.exit(1)
        for notebook in notebooks:
            process_notebook(notebook)
        return

    for notebook_path in args.notebooks:
        process_notebook(Path(notebook_path))


if __name__ == "__main__":
    main()
