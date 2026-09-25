"""Tests for frame<->timestamp conversions and sliding-window computation
(app/services/video.py)."""
from __future__ import annotations

import pytest

from app.services.video import compute_windows, frame_to_timestamp, timestamp_to_frame


def test_frame_to_timestamp_basic():
    assert frame_to_timestamp(30, fps=30.0) == pytest.approx(1.0)
    assert frame_to_timestamp(0, fps=30.0) == pytest.approx(0.0)
    assert frame_to_timestamp(15, fps=30.0) == pytest.approx(0.5)


def test_frame_to_timestamp_handles_zero_fps():
    assert frame_to_timestamp(10, fps=0.0) == 0.0


def test_timestamp_to_frame_basic():
    assert timestamp_to_frame(1.0, fps=30.0) == 30
    assert timestamp_to_frame(0.5, fps=30.0) == 15


def test_timestamp_to_frame_handles_zero_fps():
    assert timestamp_to_frame(1.0, fps=0.0) == 0


def test_frame_timestamp_round_trip():
    fps = 29.97
    for frame in [0, 1, 10, 100, 500]:
        ts = frame_to_timestamp(frame, fps)
        recovered = timestamp_to_frame(ts, fps)
        assert recovered == frame


def test_compute_windows_basic_no_remainder():
    windows = compute_windows(frame_count=16, window_size=8, window_overlap=4)
    assert windows == [(0, 8), (4, 12), (8, 16)]


def test_compute_windows_covers_trailing_remainder():
    windows = compute_windows(frame_count=20, window_size=8, window_overlap=4)
    assert windows[-1][1] == 20
    for start, end in windows:
        assert end - start <= 8
        assert end > start


def test_compute_windows_video_shorter_than_window():
    windows = compute_windows(frame_count=5, window_size=8, window_overlap=4)
    assert windows == [(0, 5)]


def test_compute_windows_zero_frames():
    assert compute_windows(frame_count=0, window_size=8, window_overlap=4) == []


def test_compute_windows_no_overlap():
    windows = compute_windows(frame_count=24, window_size=8, window_overlap=0)
    assert windows == [(0, 8), (8, 16), (16, 24)]
