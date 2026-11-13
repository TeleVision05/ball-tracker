"""The three dependency manifests must not drift apart."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _split(requirement: str) -> tuple[str, str]:
    """('numpy', '>=1.26,<3') from 'numpy>=1.26,<3'."""
    match = re.match(r"([A-Za-z0-9_.-]+(?:\[[^\]]+\])?)\s*(.*)", requirement.strip())
    assert match, requirement
    return match.group(1).lower(), match.group(2).replace(" ", "")


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _requirement_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def test_every_runtime_dependency_has_lower_and_upper_bound() -> None:
    project = _pyproject()["project"]
    backend = _requirement_lines(ROOT / "balltrace" / "backend" / "requirements.txt")
    for requirement in [*project["dependencies"], *backend]:
        name, spec = _split(requirement)
        assert ">=" in spec and "<" in spec, f"{name} needs a lower and an upper bound"


def test_shared_packages_use_identical_bounds() -> None:
    root = dict(_split(r) for r in _pyproject()["project"]["dependencies"])
    backend = dict(
        _split(r) for r in _requirement_lines(ROOT / "balltrace" / "backend" / "requirements.txt")
    )
    shared = set(root) & set(backend)
    assert {"numpy", "opencv-python", "pillow"} <= shared
    for name in shared:
        assert root[name] == backend[name], f"{name} bounds differ between manifests"


def test_requirements_txt_defers_to_pyproject() -> None:
    assert _requirement_lines(ROOT / "requirements.txt") == ["-e .[dev]"]


def test_requires_python_matches_supported_versions() -> None:
    assert _pyproject()["project"]["requires-python"] == ">=3.11,<3.13"
