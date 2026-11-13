"""Tests for trajectory segmenting/gap-handling logic."""

from __future__ import annotations

from app.services.trajectory import build_trajectory_segments, should_connect


def _point(frame_index, visible=True, x=1.0, y=1.0):
    return {"frame_index": frame_index, "ball_visible": visible, "x": x, "y": y}


def test_should_connect_within_gap():
    assert should_connect(10, 13, max_gap_frames=5) is True


def test_should_connect_exactly_at_limit():
    assert should_connect(10, 15, max_gap_frames=5) is True


def test_should_not_connect_beyond_limit():
    assert should_connect(10, 16, max_gap_frames=5) is False


def test_single_segment_for_contiguous_detections():
    detections = [_point(i) for i in range(5)]
    segments = build_trajectory_segments(detections, max_gap_frames=2)
    assert len(segments) == 1
    assert len(segments[0]) == 5


def test_breaks_segment_across_large_gap():
    detections = [_point(0), _point(1), _point(2), _point(50), _point(51)]
    segments = build_trajectory_segments(detections, max_gap_frames=5)
    assert len(segments) == 2
    assert [d["frame_index"] for d in segments[0]] == [0, 1, 2]
    assert [d["frame_index"] for d in segments[1]] == [50, 51]


def test_ignores_non_visible_detections():
    detections = [
        _point(0),
        _point(1, visible=False, x=None, y=None),
        _point(2),
    ]
    segments = build_trajectory_segments(detections, max_gap_frames=5)
    assert len(segments) == 1
    assert [d["frame_index"] for d in segments[0]] == [0, 2]


def test_ignores_detections_missing_coordinates():
    detections = [_point(0), {"frame_index": 1, "ball_visible": True, "x": None, "y": None}]
    segments = build_trajectory_segments(detections, max_gap_frames=5)
    assert len(segments) == 1
    assert len(segments[0]) == 1


def test_empty_detections_returns_no_segments():
    assert build_trajectory_segments([], max_gap_frames=5) == []


def test_unsorted_input_is_sorted_before_segmenting():
    detections = [_point(5), _point(0), _point(1)]
    segments = build_trajectory_segments(detections, max_gap_frames=5)
    assert len(segments) == 1
    assert [d["frame_index"] for d in segments[0]] == [0, 1, 5]
