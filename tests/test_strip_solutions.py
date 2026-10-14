from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_strip_solutions_generates_exercise_notebook(tmp_path: Path) -> None:
    notebook_dir = tmp_path / "notebooks" / "00_pytorch"
    notebook_dir.mkdir(parents=True)

    solution_path = notebook_dir / "solution.ipynb"
    solution_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {"language": "markdown"},
                        "source": ["# Title\n"],
                    },
                    {
                        "cell_type": "code",
                        "metadata": {"language": "python", "tags": ["solution"]},
                        "execution_count": 7,
                        "outputs": [{"output_type": "stream", "text": ["done\n"]}],
                        "source": ["x = 1\n", "print(x)\n"],
                    },
                    {
                        "cell_type": "code",
                        "metadata": {"language": "python"},
                        "execution_count": None,
                        "outputs": [],
                        "source": ["y = 2\n"],
                    },
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    script_path = Path("/workspace/scripts/strip_solutions.py")
    subprocess.run(
        [sys.executable, str(script_path), str(solution_path)],
        check=True,
        cwd="/workspace",
    )

    exercise_path = notebook_dir / "exercise.ipynb"
    assert exercise_path.exists()

    exercise = json.loads(exercise_path.read_text(encoding="utf-8"))
    assert exercise["cells"][1]["source"] == [
        "# YOUR CODE HERE\n",
        "raise NotImplementedError\n",
    ]
    assert exercise["cells"][1]["execution_count"] is None
    assert exercise["cells"][1]["outputs"] == []
    assert exercise["cells"][2]["source"] == ["y = 2\n"]
