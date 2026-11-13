"""Characterisation tests for the 2D tracker (balltracker.tracking).

These pin what the tracker does today so the module can be split without a
behaviour change. Clip-based tests use seeded synthetic deliveries and assert
against the simulator's truth with tolerances (detections, track continuity,
release and bounce frames), so they hold across OpenCV builds. The small
hand-built cases pin exact outputs of the pure helpers.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from balltracker.config import load_config
from balltracker.errors import VideoOpenError
from balltracker.release import ReleaseSeed
from balltracker.tracking import (
    TrackPoint,
    track_delivery_hybrid,
    track_video,
)
from synthetic_clips import FakeCSRT, make_clip, make_hard_clip, make_tiny_clip

SEEDS = (42, 7, 2026)


def _no_pose(**overrides) -> dict:
    config = load_config()
    config["detection"]["use_pose_release"] = False
    config["detection"].update(overrides)
    return config


def _seed_at(clip, frame_idx: int = 3) -> ReleaseSeed:
    u, v = clip.track_2d[frame_idx]
    return ReleaseSeed(frame_idx, float(u), float(v), (0.0, 1.0), 0.8)


def _errors_px(track: list[TrackPoint], clip) -> list[float]:
    return [
        float(
            np.hypot(tp.u - clip.track_2d[tp.frame_idx][0], tp.v - clip.track_2d[tp.frame_idx][1])
        )
        for tp in track
    ]


def _frames(track: list[TrackPoint]) -> list[int]:
    return [tp.frame_idx for tp in track]


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[int, object]:
    root = tmp_path_factory.mktemp("tracking_clips")
    return {seed: make_clip(root, seed) for seed in SEEDS}


@pytest.fixture(scope="module")
def hard_clip(tmp_path_factory: pytest.TempPathFactory):
    return make_hard_clip(tmp_path_factory.mktemp("hard_clip"), 2026)


# ---------------------------------------------------------------------------
# Hybrid tracker on seeded synthetic clips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_hybrid_follows_the_ball_from_release_to_bounce(clips, seed: int) -> None:
    clip = clips[seed]
    track = track_delivery_hybrid(clip.video, fps=120, config=_no_pose())
    frames = _frames(track)

    assert len(track) >= 0.75 * clip.frame_count
    assert all(not tp.predicted for tp in track)
    assert frames == sorted(set(frames)), "frames must be unique and increasing"
    assert max(np.diff(frames)) <= 5, "the ballistic filter splits on gaps over 5 frames"
    assert max(_errors_px(track, clip)) < 20.0
    assert frames[0] <= 8, "track should start near release (frame 0)"
    assert frames[-1] >= clip.frame_count - 3
    bounce_seen = max(track, key=lambda tp: tp.v).frame_idx
    assert abs(bounce_seen - clip.bounce_frame) <= 3


@pytest.mark.parametrize("seed", SEEDS)
def test_hybrid_with_release_seed_starts_at_the_seed(clips, seed: int) -> None:
    clip = clips[seed]
    track = track_delivery_hybrid(
        clip.video, fps=120, config=_no_pose(), release_seed=_seed_at(clip)
    )

    # Detection starts two frames before the seed frame (3).
    assert 1 <= track[0].frame_idx <= 3
    # Under 6 px with OpenCV 5.0; up to 16 px with OpenCV 4.10 (seed 2026).
    assert max(_errors_px(track, clip)) < 20.0
    assert len(track) >= 0.8 * clip.frame_count
    assert max(np.diff(_frames(track))) <= 6


def test_hybrid_prepends_the_seed_when_tracking_starts_later(clips) -> None:
    clip = clips[42]
    seed = _seed_at(clip, 2)
    track = track_delivery_hybrid(
        clip.video, fps=120, config=_no_pose(), release_seed=seed, start_frame=5
    )
    first = track[0]
    assert (first.frame_idx, first.u, first.v) == (2, seed.u, seed.v)
    assert first.confidence == 0.5 and first.predicted is False
    assert first.t_s == pytest.approx(2 / 120)
    assert track[1].frame_idx >= 5


def test_hybrid_window_is_respected_and_order_does_not_matter(clips) -> None:
    clip = clips[7]
    window = track_delivery_hybrid(
        clip.video, fps=120, config=_no_pose(), start_frame=10, end_frame=40
    )
    swapped = track_delivery_hybrid(
        clip.video, fps=120, config=_no_pose(), start_frame=40, end_frame=10
    )
    assert window and all(10 <= f <= 40 for f in _frames(window))
    assert window == swapped
    whole = track_delivery_hybrid(clip.video, fps=120, config=_no_pose())
    assert track_delivery_hybrid(clip.video, fps=120, config=_no_pose(), end_frame=-1) == whole


def test_hybrid_reads_fps_from_the_clip(clips) -> None:
    clip = clips[42]
    assert track_delivery_hybrid(clip.video, config=_no_pose()) == track_delivery_hybrid(
        clip.video, fps=120, config=_no_pose()
    )


def test_hybrid_ignores_a_failing_pose_seed(clips, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("pose model failed")

    monkeypatch.setattr("balltracker.tracking.estimate_release_seed", _boom)
    clip = clips[42]
    with_pose = load_config()
    assert track_delivery_hybrid(clip.video, fps=120, config=with_pose) == track_delivery_hybrid(
        clip.video, fps=120, config=_no_pose()
    )


def test_hybrid_on_a_cluttered_clip_skips_hidden_frames(hard_clip) -> None:
    clip = hard_clip
    track = track_delivery_hybrid(
        clip.video, fps=120, config=_no_pose(), release_seed=_seed_at(clip)
    )
    frames = _frames(track)
    assert not {20, 21, 22} & set(frames), "hidden frames are bridged, not invented"
    assert len(track) >= 0.75 * clip.frame_count
    assert max(_errors_px(track, clip)) < 8.0


def test_hybrid_csrt_branch(hard_clip, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stock opencv-python has no CSRT; a fake one pins the branch."""
    monkeypatch.setattr(cv2, "TrackerCSRT_create", FakeCSRT, raising=False)
    monkeypatch.setattr(FakeCSRT, "created", 0)
    clip = hard_clip
    config = _no_pose()
    track = track_delivery_hybrid(clip.video, fps=120, config=config, release_seed=_seed_at(clip))
    assert FakeCSRT.created > 0
    assert max(_errors_px(track, clip)) < 8.0

    monkeypatch.setattr(FakeCSRT, "fail_init", True)
    failed = track_delivery_hybrid(clip.video, fps=120, config=config, release_seed=_seed_at(clip))
    assert len(failed) >= 0.75 * clip.frame_count

    monkeypatch.setattr(FakeCSRT, "created", 0)
    no_csrt = _no_pose(use_csrt=False)
    track_delivery_hybrid(clip.video, fps=120, config=no_csrt, release_seed=_seed_at(clip))
    assert FakeCSRT.created == 0


def test_hybrid_on_a_clip_shorter_than_the_warm_up(tmp_path: Path) -> None:
    tiny = make_tiny_clip(tmp_path)
    track = track_delivery_hybrid(tiny, fps=120, config=_no_pose(), end_frame=50)
    assert len(track) <= 3


def test_hybrid_rejects_a_missing_video(tmp_path: Path) -> None:
    with pytest.raises(VideoOpenError):
        track_delivery_hybrid(tmp_path / "missing.mp4", fps=120, config=_no_pose())


# ---------------------------------------------------------------------------
# track_video
# ---------------------------------------------------------------------------


def test_track_video_auto_is_the_hybrid_tracker(clips) -> None:
    clip = clips[2026]
    config = _no_pose()
    assert track_video(clip.video, fps=120, config=config) == track_delivery_hybrid(
        clip.video, fps=120, config=config
    )


def test_track_video_colour_and_white_methods(clips) -> None:
    clip = clips[7]
    hsv = track_video(clip.video, fps=120, config=_no_pose(), method="hsv")
    assert len(hsv) >= 0.5 * clip.frame_count
    assert max(_errors_px(hsv, clip)) < 20.0
    assert track_video(clip.video, fps=120, config=_no_pose(), method="white") == []
    windowed = track_video(
        clip.video, fps=120, config=_no_pose(), method="hsv", start_frame=5, end_frame=45
    )
    assert windowed and all(5 <= f <= 45 for f in _frames(windowed))
