"""Unit tests for the session app's non-UI logic (balltracker.app.logic and .images)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from balltracker.app import images, logic
from balltracker.errors import VideoOpenError
from balltracker.landmarks import LANDMARK_ORDER
from synthetic_clips import make_tiny_clip


def test_app_paths(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    config = {"paths": {"raw_video": "clips/in", "sessions": "my_sessions"}}
    assert logic.app_paths(config, root) == (root / "clips/in", Path("my_sessions"))
    assert logic.app_paths({}, root) == (root / "data/raw/real", root / "sessions")


def test_list_videos(tmp_path: Path) -> None:
    for name in ("b.MOV", "a.mp4", "notes.txt", "c.avi"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in logic.list_videos(tmp_path)] == ["a.mp4", "b.MOV"]
    assert logic.list_videos(tmp_path / "missing") == []


def test_read_video_frame_clamps_and_reports_size(tmp_path: Path) -> None:
    clip = make_tiny_clip(tmp_path, n_frames=4)
    first, size = logic.read_video_frame(clip, -3)
    last, _ = logic.read_video_frame(clip, 99)
    assert size == (320, 240) and first.shape == (240, 320, 3)
    assert not np.array_equal(first, last)
    assert np.array_equal(last, logic.read_video_frame(clip, 3)[0])
    assert logic.read_video_meta(clip) == {"n": 4, "fps": 120.0}


def test_read_video_frame_rejects_a_missing_clip(tmp_path: Path) -> None:
    with pytest.raises(VideoOpenError):
        logic.read_video_frame(tmp_path / "missing.mp4", 0)


@pytest.mark.parametrize(
    ("current", "calib", "last", "expected"),
    [
        ("result", False, "d1", "setup"),
        ("result", True, None, "analyze"),
        ("result", True, "d1", "result"),
        ("setup", True, "d1", "setup"),
    ],
)
def test_initial_step(current: str, calib: bool, last: str | None, expected: str) -> None:
    assert logic.initial_step(current, calib, last) == expected


def test_step_from_label() -> None:
    for step in logic.STEP_ORDER:
        assert logic.step_from_label(logic.STEP_LABELS[step]) == step


def test_remove_last_landmark_follows_click_order() -> None:
    points = {name: np.zeros(2) for name in LANDMARK_ORDER[:3]}
    same = logic.remove_last_landmark(points)
    assert same is points and list(points) == LANDMARK_ORDER[:2]
    assert logic.remove_last_landmark({}) == {}


def test_display_to_image() -> None:
    assert logic.display_to_image({"x": 48, "y": 27}, 0.5) == (96.0, 54.0)


def test_windows() -> None:
    assert logic.fallback_window(1000, 120.0) == (0, 360)
    assert logic.fallback_window(100, 120.0) == (0, 99)
    assert logic.trim_start_here(30, 50) == (30, 50)
    assert logic.trim_start_here(60, 50) == (60, 60)
    assert logic.trim_end_here(30, 50) == (30, 50)
    assert logic.trim_end_here(30, 10) == (10, 10)
    assert logic.clip_length_text(10, 129, 120.0) == "1.00 seconds · 120 frames"


def test_suggest_window_falls_back_when_suggesting_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(_path):
        raise RuntimeError("cannot read clip")

    monkeypatch.setattr(logic, "suggest_flight_window", _boom)
    assert logic.suggest_window(tmp_path / "clip.mp4", 600, 60.0) == (0, 180)
    monkeypatch.setattr(logic, "suggest_flight_window", lambda _path: (12, 40))
    assert logic.suggest_window(tmp_path / "clip.mp4", 600, 60.0) == (12, 40)


def test_marking_helpers() -> None:
    assert logic.normalise_clicks({"3": [1.0, 2.0], 4: [3.0, 4.0]}) == {
        3: [1.0, 2.0],
        4: [3.0, 4.0],
    }
    clicks = {0: [0.0, 0.0], 2: [10.0, 4.0]}
    assert logic.click_tuples(clicks) == {0: (0.0, 0.0), 2: (10.0, 4.0)}
    assert logic.prediction_for(clicks, 5, 120.0) == (25.0, 10.0)
    assert logic.prediction_for(clicks, 2, 120.0) is None, "a marked frame needs no guess"
    assert logic.prediction_for({}, 2, 120.0) is None
    assert logic.next_mark_frame(9, 10) == 10
    assert logic.next_mark_frame(10, 10) is None
    progress = logic.marking_progress_text(2, 12, 10, 40)
    assert "**2** marked · frame **12** of **10–40** (31 frames)." in progress
    assert logic.marks_needed_text(2) == "Mark at least **5** frames (3 more) to calculate."


def test_result_helpers(tmp_path: Path) -> None:
    assert logic.default_delivery_index(["d1", "d2", "d3"], "d2") == 1
    assert logic.default_delivery_index(["d1", "d2", "d3"], "gone") == 2
    assert logic.speed_text(132.44) == "132.4 km/h"
    assert logic.speed_text(None) == "—"
    assert logic.details_markdown({"beta_s_deg": 1.5, "pitch_line_m": 0.1}) == (
        "Swing: 1.5°  \nTurn: —°  \nLine: 0.1 m  \nLength: — m"
    )


def test_replay_choices(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.mp4"
    source = tmp_path / "source.mp4"
    record = {"overlay_path": str(overlay), "source_video_path": str(source)}
    assert logic.overlay_video(record) is None and logic.source_video(record) is None
    overlay.write_bytes(b"")
    source.write_bytes(b"x")
    assert logic.overlay_video(record) is None, "an empty overlay is not playable"
    assert logic.source_video(record) == source
    overlay.write_bytes(b"x")
    assert logic.overlay_video(record) == overlay
    assert logic.overlay_video({}) is None and logic.source_video({}) is None


def test_scaled_copy_never_enlarges() -> None:
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    small, scale = images.scaled_copy(frame, 480)
    assert small.shape == (270, 480, 3) and scale == 0.5
    same, scale = images.scaled_copy(frame, 2000)
    assert same.shape == frame.shape and scale == 1.0 and same is not frame


def test_setup_picture_prompt_and_markers() -> None:
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    assert images.setup_prompt({}) == "Click: Bowler end — left stump base"
    full = {name: np.array([100.0, 100.0]) for name in LANDMARK_ORDER}
    assert images.setup_prompt(full) == "All points placed — save when ready"
    img, scale = images.annotated_display_image(
        frame, {LANDMARK_ORDER[0]: np.array([500.0, 300.0])}
    )
    pixels = np.asarray(img)
    assert img.size == (960, 540) and scale == 1.0
    assert tuple(pixels[300, 500]) == (0, 255, 0), "green cross on the point (RGB)"


def test_mark_picture_shows_prediction_only_on_unmarked_frames() -> None:
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    clicks = {4: [200.0, 200.0], 5: [220.0, 210.0]}
    with_guess, _ = images.annotated_mark_image(frame, clicks, 6, prediction=(400.0, 300.0))
    marked, _ = images.annotated_mark_image(frame, clicks, 5, prediction=(400.0, 300.0))
    ring = np.asarray(with_guess)[300, 418]
    assert tuple(ring) == (80, 80, 255), "prediction ring (RGB of BGR 255,80,80)"
    assert tuple(np.asarray(marked)[300, 418]) == (0, 0, 0)
    assert tuple(np.asarray(marked)[210, 220]) == (0, 255, 0), "current frame's mark is green"
