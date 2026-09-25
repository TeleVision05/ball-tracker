"""Tests for the pure JSON extraction/repair/validation helpers used to
parse Ollama vision model responses (app/services/ollama_client.py)."""
from __future__ import annotations

import pytest

from app.services.ollama_client import (
    extract_json_candidate,
    parse_detection_response,
    parse_json_object,
    strip_code_fences,
)

VALID_PAYLOAD = """{
"detections": [
{"frame_index": 0, "ball_visible": true, "x": 123, "y": 456, "confidence": 0.91},
{"frame_index": 1, "ball_visible": false, "x": null, "y": null, "confidence": 0.0}
]
}"""


def test_parse_valid_json():
    parsed = parse_detection_response(VALID_PAYLOAD)
    assert len(parsed.detections) == 2
    assert parsed.detections[0].frame_index == 0
    assert parsed.detections[0].ball_visible is True
    assert parsed.detections[0].x == 123
    assert parsed.detections[1].ball_visible is False
    assert parsed.detections[1].x is None


def test_parse_markdown_wrapped_json():
    wrapped = f"```json\n{VALID_PAYLOAD}\n```"
    parsed = parse_detection_response(wrapped)
    assert len(parsed.detections) == 2


def test_parse_json_with_leading_and_trailing_prose():
    wrapped = f"Sure, here is the JSON:\n{VALID_PAYLOAD}\nLet me know if you need anything else!"
    obj = parse_json_object(wrapped)
    assert "detections" in obj


def test_strip_code_fences_removes_fence_markers():
    text = "```json\n{\"a\": 1}\n```"
    assert strip_code_fences(text) == '{"a": 1}'


def test_parse_truncated_json_recovers_complete_entries():
    truncated = (
        '{"detections": ['
        '{"frame_index": 0, "ball_visible": true, "x": 10, "y": 20, "confidence": 0.8},'
        '{"frame_index": 1, "ball_visible": true, "x": 11, "y": 21, "confidence": 0.7},'
        '{"frame_index": 2, "ball_vis'  # cut off mid-object
    )
    parsed = parse_detection_response(truncated)
    assert len(parsed.detections) == 2
    assert parsed.detections[0].frame_index == 0
    assert parsed.detections[1].frame_index == 1


def test_parse_truncated_json_missing_closing_brackets():
    truncated = (
        '{"detections": ['
        '{"frame_index": 0, "ball_visible": true, "x": 10, "y": 20, "confidence": 0.8}'
    )
    parsed = parse_detection_response(truncated)
    assert len(parsed.detections) == 1
    assert parsed.detections[0].frame_index == 0


def test_parse_raises_when_no_json_present():
    with pytest.raises(ValueError):
        parse_detection_response("I cannot see a ball in these frames at all.")


def test_extract_json_candidate_raises_without_brace():
    with pytest.raises(ValueError):
        extract_json_candidate("no json here")


def test_parse_invalid_schema_raises():
    # Valid JSON, but doesn't match the expected detection schema at all.
    with pytest.raises(ValueError):
        parse_detection_response('{"unexpected": "shape"}')
