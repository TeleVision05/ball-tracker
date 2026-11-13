"""The documented API defaults must match the values the backend really uses."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app import config
from app.models import database

CONTRACT = Path(__file__).resolve().parents[2] / "API_CONTRACT.md"


def _contract_text() -> str:
    return CONTRACT.read_text(encoding="utf-8")


def _defaults_table() -> dict[str, str]:
    """Rows of the `#### Defaults` table as {field: default}."""
    section = _contract_text().split("#### Defaults", 1)[1]
    rows: dict[str, str] = {}
    for line in section.splitlines():
        match = re.match(r"\|\s*`(\w+)`\s*\|\s*`([^`]+)`\s*\|", line)
        if match:
            rows[match.group(1)] = match.group(2)
        elif rows and not line.startswith("|"):
            break
    return rows


def _settings_example() -> dict:
    """The JSON example under `### GET /api/settings/ollama`."""
    section = _contract_text().split("### GET /api/settings/ollama", 1)[1]
    block = re.search(r"```json\n(.*?)\n```", section, re.S)
    assert block, "GET /api/settings/ollama has no JSON example"
    return json.loads(block.group(1))


EXPECTED = {
    "base_url": config.DEFAULT_OLLAMA_URL,
    "model": config.DEFAULT_MODEL,
    "window_size": config.DEFAULT_WINDOW_SIZE,
    "window_overlap": config.DEFAULT_WINDOW_OVERLAP,
    "frame_stride": config.DEFAULT_FRAME_STRIDE,
}


def test_defaults_table_matches_config() -> None:
    table = _defaults_table()
    assert set(table) == set(EXPECTED)
    for field, value in EXPECTED.items():
        assert table[field] == str(value), field


def test_settings_example_matches_config() -> None:
    example = _settings_example()
    for field, value in EXPECTED.items():
        assert example[field] == value, field


def test_stored_settings_defaults_match_config() -> None:
    assert database._DEFAULT_SETTINGS == EXPECTED


def test_contract_documents_individual_frame_mode() -> None:
    """The contract says per-frame mode is on; it must really be on."""
    assert config.ANALYZE_FRAMES_INDIVIDUALLY is True
    assert "ANALYZE_FRAMES_INDIVIDUALLY = True" in _contract_text()
