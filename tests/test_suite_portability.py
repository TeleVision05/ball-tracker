"""The suite must run the way CI and the README run it.

CI installs into the runner's own Python and calls plain ``pytest tests/``.
Two things broke that while passing locally: a ``from tests.<module>`` import
(it only resolves when the repo root is on ``sys.path``, as with
``python -m pytest`` from the root) and a subprocess pinned to
``<repo>/.venv/bin/python`` (absent on a CI runner).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def test_suite_collects_without_the_repo_root_on_sys_path(tmp_path: Path) -> None:
    # `python -m pytest` adds the working directory to sys.path; running it from
    # an empty directory leaves the repo root off, exactly like plain `pytest`.
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", TESTS],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]


def test_no_test_depends_on_a_repo_local_virtualenv() -> None:
    marker = "." + "venv"
    offenders = [
        path.name
        for path in sorted(TESTS.glob("*.py"))
        if path != Path(__file__).resolve() and f'"{marker}"' in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"use sys.executable instead of a repo-local venv: {offenders}"
