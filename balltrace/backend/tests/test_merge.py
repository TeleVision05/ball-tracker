"""Tests for overlapping-window detection merge + dedupe logic."""

from __future__ import annotations

from app.services.merge import merge_detections


def _raw(frame_index, ball_visible, confidence, source_window, x=1.0, y=2.0):
    return {
        "frame_index": frame_index,
        "ball_visible": ball_visible,
        "x": x if ball_visible else None,
        "y": y if ball_visible else None,
        "confidence": confidence,
        "source_window": source_window,
    }


def test_dedupes_same_frame_across_windows():
    raw = [
        _raw(5, True, 0.6, source_window=0),
        _raw(5, True, 0.6, source_window=1),
    ]
    merged = merge_detections(raw)
    assert len(merged) == 1
    assert merged[0]["frame_index"] == 5


def test_prefers_higher_confidence_when_both_visible():
    raw = [
        _raw(10, True, 0.4, source_window=0),
        _raw(10, True, 0.9, source_window=1),
    ]
    merged = merge_detections(raw)
    assert len(merged) == 1
    assert merged[0]["confidence"] == 0.9
    assert merged[0]["source_window"] == 1


def test_prefers_visible_over_not_visible_regardless_of_confidence():
    raw = [
        _raw(3, False, 0.99, source_window=0),
        _raw(3, True, 0.2, source_window=1),
    ]
    merged = merge_detections(raw)
    assert len(merged) == 1
    assert merged[0]["ball_visible"] is True
    assert merged[0]["confidence"] == 0.2


def test_prefers_higher_confidence_when_both_not_visible():
    raw = [
        _raw(7, False, 0.1, source_window=0),
        _raw(7, False, 0.5, source_window=1),
    ]
    merged = merge_detections(raw)
    assert len(merged) == 1
    assert merged[0]["confidence"] == 0.5


def test_merges_multiple_overlapping_windows_and_sorts_by_frame():
    raw = [
        _raw(4, True, 0.5, source_window=0),
        _raw(3, True, 0.5, source_window=0),
        _raw(3, True, 0.9, source_window=1),  # overlap frame, better confidence
        _raw(5, False, 0.0, source_window=1),
        _raw(6, True, 0.7, source_window=1),
    ]
    merged = merge_detections(raw)
    assert [d["frame_index"] for d in merged] == [3, 4, 5, 6]
    assert merged[0]["confidence"] == 0.9  # frame 3 picked the better window


def test_empty_input_returns_empty_list():
    assert merge_detections([]) == []
