"""Tests for normalized coordinate computation and full detection record
construction (app/services/analysis_pipeline.py)."""

from __future__ import annotations

import pytest

from app.services.analysis_pipeline import build_detection_record, to_normalized


def test_to_normalized_basic():
    x, y = to_normalized(640.0, 360.0, width=1920, height=1080)
    assert x == pytest.approx(640 / 1920)
    assert y == pytest.approx(360 / 1080)


def test_to_normalized_returns_none_when_no_coordinates():
    x, y = to_normalized(None, None, width=1920, height=1080)
    assert x is None
    assert y is None


def test_to_normalized_returns_none_when_dimensions_unknown():
    x, y = to_normalized(100.0, 100.0, width=0, height=0)
    assert x is None
    assert y is None


def test_build_detection_record_visible():
    raw = {
        "frame_index": 42,
        "ball_visible": True,
        "x": 640.0,
        "y": 360.0,
        "confidence": 0.91,
        "source_window": 5,
    }
    record = build_detection_record(raw, fps=30.0, width=1920, height=1080)
    assert record["frame_index"] == 42
    assert record["timestamp"] == 42 / 30.0
    assert record["ball_visible"] is True
    assert record["normalized_x"] == 640.0 / 1920
    assert record["normalized_y"] == 360.0 / 1080
    assert record["confidence"] == 0.91
    assert record["source_window"] == 5
    assert "id" in record


def test_build_detection_record_not_visible_has_no_coordinates():
    raw = {
        "frame_index": 10,
        "ball_visible": False,
        "x": None,
        "y": None,
        "confidence": 0.0,
        "source_window": 1,
    }
    record = build_detection_record(raw, fps=30.0, width=1920, height=1080)
    assert record["ball_visible"] is False
    assert record["x"] is None
    assert record["normalized_x"] is None
    assert record["normalized_y"] is None
