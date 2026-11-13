"""Tests for contact-sheet construction and coordinate remapping."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.analysis_pipeline import sample_frame_indices
from app.services.video import ContactSheet, build_contact_sheet, compute_windows


def test_build_contact_sheet_dimensions():
    frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(4)]
    sheet = build_contact_sheet(frames, panel_width=160, quality=80)
    assert sheet.frame_count == 4
    assert sheet.panel_width == 160
    assert sheet.panel_height == 90
    assert sheet.original_width == 320
    assert sheet.original_height == 180
    assert len(sheet.jpeg_bytes) > 0


def test_scale_to_original():
    sheet = ContactSheet(
        jpeg_bytes=b"",
        panel_width=160,
        panel_height=90,
        original_width=320,
        original_height=180,
        frame_count=2,
    )
    ox, oy = sheet.scale_to_original(80.0, 45.0)
    assert ox == pytest.approx(160.0)
    assert oy == pytest.approx(90.0)


def test_compute_windows_with_remainder():
    windows = compute_windows(frame_count=10, window_size=4, window_overlap=2)
    assert windows[0] == (0, 4)
    assert windows[-1][1] == 10
    # overlapping coverage of every frame
    covered = set()
    for start, end in windows:
        covered.update(range(start, end))
    assert covered == set(range(10))


def test_sample_frame_indices_stride():
    assert sample_frame_indices(10, 3) == [0, 3, 6, 9]
    assert sample_frame_indices(11, 3) == [0, 3, 6, 9, 10]
    assert sample_frame_indices(5, 1) == [0, 1, 2, 3, 4]
