"""CLI test for analyzing one Android capture sidecar."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tests.test_capture_analysis import _write_synthetic_capture

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SCRIPT = ROOT / "scripts" / "analyze_capture.py"


def test_analyze_capture_cli_writes_report(tmp_path: Path) -> None:
    sidecar_path, _gt = _write_synthetic_capture(tmp_path / "clip")
    out_path = tmp_path / "analysis.json"
    completed = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--sidecar",
            str(sidecar_path),
            "--method",
            "hsv",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert out_path.exists()
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert "release_speed_kmh" in result["metrics"]
    assert "Wrote analysis" in completed.stdout


def test_analyze_capture_cli_missing_sidecar(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--sidecar",
            str(tmp_path / "missing.json"),
            "--out",
            str(tmp_path / "analysis.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert completed.stderr.startswith("Error:")
