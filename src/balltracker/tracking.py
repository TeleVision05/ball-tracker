"""2D ball tracking across video frames.

This is the public entry point. The pieces live in focused modules and are
re-exported here, so ``from balltracker.tracking import ...`` keeps working:

- :mod:`balltracker.tracking_core`: ``TrackPoint``, the Kalman filter,
  ``associate_detections`` and ``filter_ballistic_segment``.
- :mod:`balltracker.tracking_hybrid`: the hybrid tracker's per-frame detection
  (gated MOG2 motion, colour fallback, CSRT).
- :mod:`balltracker.tracking_manual`: ``predict_ball_uv`` and
  ``track_from_clicks`` for manual marking.

``estimate_release_seed`` is looked up on this module, so tests can patch
``balltracker.tracking.estimate_release_seed``.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from .config import load_config
from .detection import BallDetection, detect_video
from .errors import VideoOpenError
from .release import ReleaseSeed, estimate_release_seed
from .tracking_core import (
    TrackPoint,
    _Kalman2D,
    associate_detections,
    filter_ballistic_segment,
)
from .tracking_hybrid import _create_csrt_tracker, detect_hybrid
from .tracking_manual import predict_ball_uv, track_from_clicks

__all__ = [
    "BallDetection",
    "ReleaseSeed",
    "TrackPoint",
    "_Kalman2D",
    "_create_csrt_tracker",
    "associate_detections",
    "estimate_release_seed",
    "filter_ballistic_segment",
    "predict_ball_uv",
    "track_delivery_hybrid",
    "track_from_clicks",
    "track_video",
]


def _clamp_window(start_frame: int, end_frame: int | None, total: int) -> tuple[int, int]:
    """Clamp the frame window to the clip; a negative or missing end means the last frame."""
    if end_frame is None or end_frame < 0:
        end_frame = max(0, total - 1)
    start_frame = int(max(0, start_frame))
    end_frame = int(min(end_frame, max(0, total - 1)))
    if end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame
    return start_frame, end_frame


def _prepend_seed(
    track: list[TrackPoint], release_seed: ReleaseSeed | None, fps: float
) -> list[TrackPoint]:
    """Start the track at the release seed when detection picked the ball up later."""
    if release_seed is None or not track or track[0].frame_idx <= release_seed.frame_idx:
        return track
    seed_point = TrackPoint(
        frame_idx=release_seed.frame_idx,
        t_s=release_seed.frame_idx / max(fps, 1e-6),
        u=release_seed.u,
        v=release_seed.v,
        radius_px=track[0].radius_px,
        confidence=0.5,
        predicted=False,
    )
    return [seed_point] + track


def track_delivery_hybrid(
    video_path: str | Path,
    fps: float | None = None,
    config: dict | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    release_seed: ReleaseSeed | None = None,
) -> list[TrackPoint]:
    """Blur-tolerant hybrid tracker: pose-seeded gate + MOG2 motion + CSRT.

    Combines a MediaPipe pose-estimated release point (when available) with
    a gated MOG2 background-subtraction search, a small colour-detection
    fallback inside the gate, and a CSRT visual tracker once locked on — so
    nets footage with motion blur and cluttered backgrounds can still be
    tracked, not just clean synthetic/red-ball clips.
    """
    if config is None:
        config = load_config()
    det_cfg = config.get("detection", {})

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoOpenError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_val = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps is None:
        fps = fps_val if fps_val > 1 else float(config["camera"]["fps"])
    start_frame, end_frame = _clamp_window(start_frame, end_frame, total)

    if release_seed is None and det_cfg.get("use_pose_release", True):
        try:
            release_seed = estimate_release_seed(
                video_path, start_frame=start_frame, end_frame=end_frame, config=config
            )
        except Exception:
            release_seed = None

    detections = detect_hybrid(cap, total, start_frame, end_frame, release_seed, config)
    cap.release()

    track = associate_detections(
        detections,
        fps=fps,
        max_jump_px=float(det_cfg.get("hybrid_max_jump_px", 55.0)),
        early_max_jump_px=float(det_cfg.get("hybrid_early_max_jump_px", 90.0)),
    )
    return filter_ballistic_segment(_prepend_seed(track, release_seed, fps))


def track_video(
    video_path: str | Path,
    fps: float | None = None,
    config: dict | None = None,
    method: str = "auto",
    max_jump_px: float = 55.0,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> list[TrackPoint]:
    """Detect and track the ball; optionally limit to a frame window."""
    if config is None:
        config = load_config()
    if fps is None:
        cap = cv2.VideoCapture(str(video_path))
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        fps = float(fps_val) if fps_val and fps_val > 1 else float(config["camera"]["fps"])

    if method == "auto":
        return track_delivery_hybrid(
            video_path,
            fps=fps,
            config=config,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    detections = detect_video(
        video_path,
        config=config,
        method=method,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    track = associate_detections(detections, fps=fps, max_jump_px=max_jump_px)
    return filter_ballistic_segment(track)
