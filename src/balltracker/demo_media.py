"""Overlay video and pitch map for ``balltracker demo``."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .calibration import CameraCalibration, project_points
from .demo_scene import (
    VideoWriteError,
    draw_hud,
    draw_overlay,
    ffmpeg_available,
    open_writer,
    video_codec,
)
from .physics import TrajectoryFit, propagate
from .pitch import STUMP_WIDTH_M
from .plain_language import _FULL_MAX_M, _GOOD_MAX_M, _SHORT_MAX_M, _YORKER_MAX_M
from .viz import ensure_web_playable

LENGTH_BANDS = (
    ("yorker", 0.0, _YORKER_MAX_M, "#f3e3c3"),
    ("full", _YORKER_MAX_M, _FULL_MAX_M, "#e9d3a6"),
    ("good length", _FULL_MAX_M, _GOOD_MAX_M, "#d9bf8a"),
    ("short", _GOOD_MAX_M, _SHORT_MAX_M, "#c9ab72"),
    ("too short", _SHORT_MAX_M, 12.0, "#b8975c"),
)


@dataclass(frozen=True)
class OverlayPlan:
    """Everything the overlay needs, in video-frame terms."""

    track_frames: np.ndarray  # (N,) frame index of each detection
    track_uv: np.ndarray  # (N, 2) detected pixel centres
    fit_uv_by_frame: dict[int, tuple[float, float]]
    bounce_frame: int | None
    bounce_uv: tuple[float, float] | None
    flight_lines: list[str]
    result_lines: list[str]
    view: tuple[int, int, int, int]  # x0, y0, width, height of the crop shown


def view_around(points_uv: np.ndarray, frame_size: tuple[int, int], aspect: float) -> tuple:
    """A crop with the given aspect that holds every point plus a margin."""
    w, h = frame_size
    lo, hi = points_uv.min(axis=0), points_uv.max(axis=0)
    centre = (lo + hi) / 2.0
    crop_h = max(h / 2.0, (hi[1] - lo[1]) * 1.9, (hi[0] - lo[0]) * 1.9 / aspect)
    crop_h = min(crop_h, h, w / aspect)
    crop_w = crop_h * aspect
    x0 = float(np.clip(centre[0] - crop_w / 2.0, 0, w - crop_w))
    y0 = float(np.clip(centre[1] - crop_h * 0.45, 0, h - crop_h))
    return int(x0), int(y0), int(crop_w), int(crop_h)


def fitted_positions(fit: TrajectoryFit, times_s: np.ndarray, g: float = 9.81) -> np.ndarray:
    """Evaluate the fitted pre- and post-bounce segments at absolute times."""
    out = np.zeros((len(times_s), 3), dtype=np.float64)
    bounce_t = fit.bounce_time_s if fit.bounce_time_s is not None else float("inf")
    for i, t in enumerate(times_s):
        segment = fit.pre if (t <= bounce_t or fit.post is None) else fit.post
        out[i] = propagate(segment, np.array([t - segment.t0]), g=g)[0]
    return out


def fitted_pixels_by_frame(
    fit: TrajectoryFit, calib: CameraCalibration, first: int, last: int, fps: float
) -> dict[int, tuple[float, float]]:
    """Projected fitted path for each video frame from ``first`` to ``last``."""
    frames = np.arange(first, last + 1)
    uv = project_points(calib, fitted_positions(fit, frames / fps))
    return {int(f): (float(u), float(v)) for f, (u, v) in zip(frames, uv, strict=True)}


def _overlay_frame(
    frame: np.ndarray, k: int, plan: OverlayPlan, size: tuple[int, int]
) -> np.ndarray:
    """Draw at source resolution, crop to the view, scale to ``size``, then add the HUD."""
    seen = plan.track_frames <= k
    fit_uv = np.array(
        [uv for f, uv in sorted(plan.fit_uv_by_frame.items()) if f <= k], dtype=np.float64
    ).reshape(-1, 2)
    after_bounce = plan.bounce_frame is not None and k >= plan.bounce_frame
    drawn = draw_overlay(
        frame, plan.track_uv[seen], fit_uv, plan.bounce_uv if after_bounce else None, []
    )
    x0, y0, cw, ch = plan.view
    out = cv2.resize(drawn[y0 : y0 + ch, x0 : x0 + cw], size, interpolation=cv2.INTER_AREA)
    draw_hud(out, plan.flight_lines)
    return out


def render_overlay_video(
    clip_path: Path,
    out_path: Path,
    plan: OverlayPlan,
    out_fps: float = 30.0,
    size: tuple[int, int] = (1280, 720),
    hold_s: float = 4.0,
) -> Path:
    """Write the overlay at ``out_fps`` (slow motion for a high-speed clip), then hold.

    The final frame is held for ``hold_s`` seconds with the results panel.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise VideoWriteError(str(clip_path))
    writer = open_writer(out_path, out_fps, size)
    raw_last: np.ndarray | None = None
    k = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_last = frame
            writer.write(_overlay_frame(frame, k, plan, size))
            k += 1
        if raw_last is not None:
            # Hold the last frame with the whole track and the results panel.
            held = _overlay_frame(raw_last, k, replace(plan, flight_lines=plan.result_lines), size)
            for _ in range(int(round(hold_s * out_fps))):
                writer.write(held)
    finally:
        writer.release()
        cap.release()
    return out_path


def make_web_playable(path: Path) -> tuple[bool, str | None]:
    """Re-encode to H.264/yuv420p when ffmpeg exists. Returns (is_h264, codec)."""
    if not ffmpeg_available():
        return False, None
    ensure_web_playable(path)
    codec = video_codec(path)
    return codec == "h264", codec


def _draw_pitch_end(ax) -> None:
    """Length bands, the batter's stumps and the popping crease, seen from above."""
    for label, start, end, colour in LENGTH_BANDS:
        ax.add_patch(Rectangle((start, -1.52), end - start, 3.04, color=colour, zorder=0))
        ax.text((start + end) / 2, 1.38, label, ha="center", va="top", fontsize=9)
    ax.axvline(1.22, color="white", linewidth=2, zorder=1)
    half = STUMP_WIDTH_M / 2.0
    ax.plot([0, 0, 0], [-half, 0, half], "s", color="#5b3a1a", markersize=5, zorder=3)


def save_pitch_map(estimate: dict, truth: dict, out_path: Path) -> Path:
    """Batter's end from above, with the estimated and the true bounce."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _draw_pitch_end(ax)
    est = (float(estimate["bounce_length_m"]), float(estimate["bounce_line_m"]))
    tru = (float(truth["bounce_length_m"]), float(truth["bounce_line_m"]))
    ax.plot(*tru, "o", ms=13, mfc="none", mec="black", mew=2, label="true bounce (simulator)")
    ax.plot(*est, "x", ms=13, mew=3, color="#d62728", label="estimated bounce")
    ax.set_xlim(-0.5, 12.0)
    ax.set_ylim(-1.52, 1.52)
    ax.set_xlabel("Distance in front of the batter's stumps (m)")
    ax.set_ylabel("Line (m)\noff side +")
    ax.set_title(
        f"Where it pitched: estimate {est[0]:.2f} m, {est[1] * 100:+.0f} cm; "
        f"truth {tru[0]:.2f} m, {tru[1] * 100:+.0f} cm (synthetic delivery)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
