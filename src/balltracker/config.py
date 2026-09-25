"""Load project configuration from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """Load BallTracker configuration dict."""
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
