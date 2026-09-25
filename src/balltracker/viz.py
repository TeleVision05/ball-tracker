"""Visualization helpers for ball tracks, pitch maps, and overlay videos."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .pitch import PITCH_LENGTH_M, STUMP_HEIGHT_M, STUMP_WIDTH_M

PITCH_CORRIDOR_M = 3.0
CLIP_MAX_SECONDS = 12.0
CLIP_PAD_FRAMES = 30


def ensure_web_playable(video_path: str | Path) -> Path:
    """Re-encode to H.264/yuv420p so Streamlit / browsers can play the file.

    OpenCV's default ``mp4v`` (MPEG-4 Part 2) often fails in ``st.video``.
    """
    video_path = Path(video_path)
    if not video_path.exists() or video_path.stat().st_size == 0:
        return video_path

    if shutil.which("ffmpeg") is None:
        return video_path

    # Skip if already H.264.
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip() in {"h264", "avc1", "hev1", "h265"}:
            return video_path
    except OSError:
        pass

    tmp = video_path.with_suffix(".h264.tmp.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(video_path)
    else:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return video_path


def _frame_idx(point: Any) -> int:
    if isinstance(point, dict):
        return int(point["frame_idx"])
    return int(point.frame_idx)


def _uv(point: Any) -> tuple[float, float]:
    if isinstance(point, dict):
        return float(point["u"]), float(point["v"])
    return float(point.u), float(point.v)


def _track_by_frame(track_list: list[Any]) -> dict[int, tuple[float, float]]:
    """Map frame index to (u, v), keeping the last point per frame."""
    out: dict[int, tuple[float, float]] = {}
    for pt in track_list:
        out[_frame_idx(pt)] = _uv(pt)
    return out


def draw_track_overlay(
    frame_bgr: np.ndarray,
    track_points_uv: list[Any],
    bounce_uv: tuple[float, float] | None = None,
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Draw polyline track and optional bounce marker on a BGR frame."""
    frame = frame_bgr.copy()
    if not track_points_uv:
        return frame

    pts = []
    for pt in track_points_uv:
        u, v = _uv(pt)
        if np.isfinite(u) and np.isfinite(v):
            pts.append((int(round(u)), int(round(v))))

    if len(pts) >= 2:
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)
    elif len(pts) == 1:
        cv2.circle(frame, pts[0], 4, color, -1, cv2.LINE_AA)

    if bounce_uv is not None:
        bu, bv = bounce_uv
        if np.isfinite(bu) and np.isfinite(bv):
            centre = (int(round(bu)), int(round(bv)))
            cv2.drawMarker(
                frame,
                centre,
                (0, 0, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=14,
                thickness=2,
            )

    return frame


def _bounce_xy_from_input(metrics_or_bounce_xy: Any) -> tuple[float, float, dict | None]:
    """Resolve bounce (line, y) and optional metrics dict for titling."""
    if isinstance(metrics_or_bounce_xy, dict):
        if "pitch_line_m" in metrics_or_bounce_xy and "bounce_y_m" in metrics_or_bounce_xy:
            return (
                float(metrics_or_bounce_xy["pitch_line_m"]),
                float(metrics_or_bounce_xy["bounce_y_m"]),
                metrics_or_bounce_xy,
            )
        if "metrics" in metrics_or_bounce_xy:
            m = metrics_or_bounce_xy["metrics"]
            return float(m["pitch_line_m"]), float(m["bounce_y_m"]), m
        if "bounce_position_m" in metrics_or_bounce_xy:
            bp = metrics_or_bounce_xy["bounce_position_m"]
            return float(bp[0]), float(bp[1]), metrics_or_bounce_xy
    if isinstance(metrics_or_bounce_xy, (tuple, list)) and len(metrics_or_bounce_xy) >= 2:
        return float(metrics_or_bounce_xy[0]), float(metrics_or_bounce_xy[1]), None
    # DeliveryMetrics-like dataclass
    if hasattr(metrics_or_bounce_xy, "pitch_line_m"):
        return (
            float(metrics_or_bounce_xy.pitch_line_m),
            float(metrics_or_bounce_xy.bounce_y_m),
            metrics_or_bounce_xy,
        )
    raise TypeError("metrics_or_bounce_xy must be metrics dict, analysis dict, or (x, y) tuple")


def draw_pitch_map(
    metrics_or_bounce_xy: Any,
    figsize: tuple[float, float] = (6, 8),
) -> Figure:
    """Top-down pitch map with stumps, bounce mark, and optional metrics title."""
    line_m, bounce_y_m, metrics = _bounce_xy_from_input(metrics_or_bounce_xy)
    half_corridor = PITCH_CORRIDOR_M / 2.0
    half_stump = STUMP_WIDTH_M / 2.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xlim(-half_corridor, half_corridor)
    ax.set_ylim(-0.5, PITCH_LENGTH_M + 0.5)
    ax.set_xlabel("Line (m, off-side +)")
    ax.set_ylabel("Length (m, from bowler)")
    ax.grid(True, alpha=0.3)

    # Pitch corridor
    rect = plt.Rectangle(
        (-half_corridor, 0.0),
        PITCH_CORRIDOR_M,
        PITCH_LENGTH_M,
        fill=False,
        edgecolor="green",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(rect)

    # Stumps at both ends
    for y_end in (0.0, PITCH_LENGTH_M):
        stump_rect = plt.Rectangle(
            (-half_stump, y_end - 0.05),
            STUMP_WIDTH_M,
            STUMP_HEIGHT_M * 0.15,
            fill=True,
            facecolor="saddlebrown",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_patch(stump_rect)

    # Bounce mark
    ax.plot(
        line_m,
        bounce_y_m,
        marker="x",
        markersize=12,
        markeredgewidth=2.5,
        color="red",
        linestyle="none",
        label="bounce",
    )
    ax.scatter([line_m], [bounce_y_m], s=40, c="red", zorder=5)

    title_parts: list[str] = []
    if metrics is not None:
        m = metrics if isinstance(metrics, dict) else metrics
        speed = getattr(m, "release_speed_kmh", None) or (
            m.get("release_speed_kmh") if isinstance(m, dict) else None
        )
        beta = getattr(m, "beta_s_deg", None) or (
            m.get("beta_s_deg") if isinstance(m, dict) else None
        )
        turn = getattr(m, "turn_deg", None) or (m.get("turn_deg") if isinstance(m, dict) else None)
        if speed is not None:
            title_parts.append(f"{float(speed):.1f} km/h")
        if beta is not None:
            title_parts.append(f"swing {float(beta):+.1f}°")
        if turn is not None:
            title_parts.append(f"turn {float(turn):+.1f}°")
    if title_parts:
        ax.set_title(" | ".join(title_parts))

    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _clip_frame_range(
    track_list: list[Any],
    total_frames: int,
    fps: float,
) -> tuple[int, int]:
    """Return inclusive frame range for overlay clip."""
    if not track_list:
        return 0, max(0, total_frames - 1)

    frames = [_frame_idx(p) for p in track_list]
    start = max(0, min(frames) - CLIP_PAD_FRAMES)
    end = min(total_frames - 1, max(frames) + CLIP_PAD_FRAMES)
    duration_s = (end - start + 1) / max(fps, 1e-6)
    if duration_s <= CLIP_MAX_SECONDS:
        return start, end

    # Long capture: keep a window centred on bounce or track midpoint.
    bounce_candidates = [f for f in frames if f >= start and f <= end]
    centre = bounce_candidates[len(bounce_candidates) // 2] if bounce_candidates else (start + end) // 2
    half = int(CLIP_MAX_SECONDS * fps / 2)
    start = max(0, centre - half)
    end = min(total_frames - 1, centre + half)
    return start, end


def render_overlay_video(
    video_path: str | Path,
    track_list: list[Any],
    out_path: str | Path,
    bounce_frame: int | None = None,
    fit_positions_projected: list[tuple[float, float]] | None = None,
    clip: bool = True,
) -> Path:
    """Burn track overlay onto video; optionally clip long captures."""
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    track_by_frame = _track_by_frame(track_list)
    sorted_frames = sorted(track_by_frame)

    if bounce_frame is None and track_list:
        # Heuristic: frame with largest v (or from analysis fit) — use mid-track min v proxy
        bounce_frame = sorted_frames[len(sorted_frames) // 2]

    bounce_uv: tuple[float, float] | None = None
    if bounce_frame is not None and bounce_frame in track_by_frame:
        bounce_uv = track_by_frame[bounce_frame]

    if clip:
        start_frame, end_frame = _clip_frame_range(track_list, total_frames, fps)
    else:
        start_frame, end_frame = 0, max(0, total_frames - 1)

    # Write intermediate with OpenCV, then transcode to H.264 for the browser.
    tmp_raw = Path(
        tempfile.mkstemp(prefix="balltracker_overlay_", suffix=".mp4")[1]
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_raw), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        tmp_raw.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    cumulative: list[tuple[int, int, int]] = []
    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        if frame_idx > end_frame:
            break

        for fi in sorted_frames:
            if fi <= frame_idx and (not cumulative or cumulative[-1][0] < fi):
                u, v = track_by_frame[fi]
                cumulative.append((fi, int(round(u)), int(round(v))))

        overlay_pts = [{"frame_idx": fi, "u": float(u), "v": float(v)} for fi, u, v in cumulative]
        out_frame = draw_track_overlay(
            frame,
            overlay_pts,
            bounce_uv=bounce_uv if bounce_frame is not None and frame_idx >= bounce_frame else None,
        )

        if fit_positions_projected:
            fit_pts = [
                (int(round(u)), int(round(v)))
                for u, v in fit_positions_projected[: frame_idx + 1]
                if np.isfinite(u) and np.isfinite(v)
            ]
            if len(fit_pts) >= 2:
                for i in range(1, len(fit_pts)):
                    cv2.line(out_frame, fit_pts[i - 1], fit_pts[i], (255, 180, 0), 1, cv2.LINE_AA)

        writer.write(out_frame)
        written += 1
        frame_idx += 1

    writer.release()
    cap.release()

    if written == 0 or not tmp_raw.exists() or tmp_raw.stat().st_size == 0:
        tmp_raw.unlink(missing_ok=True)
        shutil.copy2(video_path, out_path)
        return ensure_web_playable(out_path)

    shutil.move(str(tmp_raw), str(out_path))
    return ensure_web_playable(out_path)


def save_pitch_map_png(metrics_or_bounce_xy: Any, out_path: str | Path) -> Path:
    """Render pitch map figure to PNG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = draw_pitch_map(metrics_or_bounce_xy)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
