"""Non-UI logic of the session app: video access, routing, windows and text.

Nothing here imports Streamlit, so all of it is unit-tested directly.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..errors import VideoOpenError
from ..flight_window import suggest_flight_window
from ..tracking import predict_ball_uv
from ..landmarks import LANDMARK_ORDER

VIDEO_SUFFIXES = {".mp4", ".mov"}
STEP_ORDER = ["setup", "analyze", "result"]
STEP_LABELS = {
    "setup": "1. Set up camera",
    "analyze": "2. Analyze a ball",
    "result": "3. See result",
}
MIN_MARKED_FRAMES = 5
NO_VALUE = "—"


# ---------------------------------------------------------------------------
# Paths and video access
# ---------------------------------------------------------------------------


def app_paths(config: dict, root: Path) -> tuple[Path, Path]:
    """Default video folder (under ``root``) and sessions folder from config.

    A relative ``paths.sessions`` value is relative to the working directory,
    as it always has been.
    """
    paths = config.get("paths", {})
    default_real_dir = root / paths.get("raw_video", "data/raw/real")
    sessions_root = Path(paths.get("sessions", root / "sessions"))
    return default_real_dir, sessions_root


def list_videos(folder: Path) -> list[Path]:
    """The .mp4 and .mov files directly in ``folder``, sorted; empty if it is missing."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def read_video_frame(video_path: Path, frame_idx: int) -> tuple[np.ndarray, tuple[int, int]]:
    """One BGR frame (index clamped to the clip) and the clip's (width, height)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoOpenError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = int(np.clip(frame_idx, 0, max(0, n - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ok or frame is None:
        raise VideoOpenError(f"Cannot read frame {frame_idx}")
    return frame, (w, h)


def read_video_meta(video_path: Path) -> dict:
    """Frame count and fps; 0 frames and 30 fps when the clip cannot say."""
    cap = cv2.VideoCapture(str(video_path))
    meta = {
        "n": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
    }
    cap.release()
    return meta


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def initial_step(current: str, calib_exists: bool, last_delivery_id: str | None) -> str:
    """Where the first run lands: setup without a calibration, else analyze."""
    if not calib_exists:
        return "setup"
    if last_delivery_id is None:
        return "analyze"
    return current


def step_from_label(label: str) -> str:
    """The step key for a step-picker label."""
    return STEP_ORDER[[STEP_LABELS[s] for s in STEP_ORDER].index(label)]


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------


def remove_last_landmark(points: dict) -> dict:
    """Drop the most recent landmark (by click order) from ``points`` in place."""
    for name in reversed(LANDMARK_ORDER):
        if name in points:
            del points[name]
            break
    return points


def display_to_image(click: dict, scale: float) -> tuple[float, float]:
    """A click on the scaled-down picture, in original image pixels."""
    return float(click["x"]) / scale, float(click["y"]) / scale


# ---------------------------------------------------------------------------
# Trimming the clip
# ---------------------------------------------------------------------------


def fallback_window(n_frames: int, fps: float) -> tuple[int, int]:
    """The first three seconds, used when the flight cannot be found."""
    return 0, min(n_frames - 1, int(fps * 3))


def suggest_window(video_path: Path, n_frames: int, fps: float) -> tuple[int, int]:
    """The suggested ball-flight window, or the fallback if suggesting fails."""
    try:
        return suggest_flight_window(video_path)
    except Exception:
        return fallback_window(n_frames, fps)


def trim_start_here(preview_frame: int, end_frame: int) -> tuple[int, int]:
    """Window starting at the preview frame; the end moves up if it was before it."""
    return int(preview_frame), int(max(preview_frame, end_frame))


def trim_end_here(start_frame: int, preview_frame: int) -> tuple[int, int]:
    """Window ending at the preview frame; the start moves back if it was after it."""
    return int(min(start_frame, preview_frame)), int(preview_frame)


def clip_length_text(start_frame: int, end_frame: int, fps: float) -> str:
    frames = end_frame - start_frame + 1
    return f"{frames / fps:.2f} seconds · {frames} frames"


# ---------------------------------------------------------------------------
# Marking the ball
# ---------------------------------------------------------------------------


def normalise_clicks(clicks: dict) -> dict[int, list[float]]:
    """Clicks keyed by int frame (Streamlit may revive the keys as strings)."""
    return {int(k): v for k, v in clicks.items()}


def click_tuples(clicks: dict) -> dict[int, tuple[float, float]]:
    return {int(f): (float(uv[0]), float(uv[1])) for f, uv in clicks.items()}


def prediction_for(clicks: dict, frame: int, fps: float) -> tuple[float, float] | None:
    """Where the ball should be on an unmarked frame, from the marks so far."""
    if len(clicks) >= 1 and frame not in clicks:
        return predict_ball_uv(click_tuples(clicks), frame, fps)
    return None


def next_mark_frame(frame: int, end_frame: int) -> int | None:
    """The frame after ``frame``, or None at the end of the window."""
    nxt = int(frame) + 1
    return nxt if nxt <= int(end_frame) else None


def marking_progress_text(n_clicked: int, frame: int, start: int, end: int) -> str:
    return (
        f"Click the **ball** on each frame. A red ring shows where the track "
        f"predicts it should be from your previous clicks. "
        f"**{n_clicked}** marked · frame **{frame}** of **{start}–{end}** "
        f"({end - start + 1} frames)."
    )


def marks_needed_text(n_clicked: int) -> str:
    return (
        f"Mark at least **{MIN_MARKED_FRAMES}** frames "
        f"({MIN_MARKED_FRAMES - n_clicked} more) to calculate."
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


def default_delivery_index(delivery_ids: list[str], default_id: str | None) -> int:
    """Index of ``default_id`` in the picker, else the latest delivery."""
    if default_id in delivery_ids:
        return delivery_ids.index(default_id)
    return len(delivery_ids) - 1


def speed_text(speed_kmh: float | None) -> str:
    return f"{speed_kmh:.1f} km/h" if speed_kmh is not None else NO_VALUE


def details_markdown(metrics: dict) -> str:
    return (
        f"Swing: {metrics.get('beta_s_deg', NO_VALUE)}°  \n"
        f"Turn: {metrics.get('turn_deg', NO_VALUE)}°  \n"
        f"Line: {metrics.get('pitch_line_m', NO_VALUE)} m  \n"
        f"Length: {metrics.get('pitch_length_m', NO_VALUE)} m"
    )


def overlay_video(record: dict) -> Path | None:
    """The saved overlay replay, if it exists and is not empty."""
    overlay = record.get("overlay_path")
    path = Path(overlay) if overlay else None
    if path is not None and path.exists() and path.stat().st_size > 0:
        return path
    return None


def source_video(record: dict) -> Path | None:
    """The original clip, if it is still on disk."""
    src = record.get("source_video_path")
    path = Path(src) if src else None
    if path is not None and path.exists():
        return path
    return None
