"""Suggest a delivery flight window from raw video motion energy.

Used to auto-trim nets footage (walk-up, idle standing) down to roughly the
release-to-bounce-and-beyond window before running detection/tracking, so
noisy motion outside the actual delivery doesn't pollute the track.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import load_config

# Central pitch corridor: excludes side clutter (fielders, sightscreen edges)
# and top/bottom borders (bowler's head, footmarks) while keeping the flight line.
_CORRIDOR_X = (0.10, 0.90)
_CORRIDOR_Y = (0.15, 0.85)


def motion_energy_series(
    video_path: str | Path,
    sample_stride: int = 2,
    config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Compute per-sampled-frame motion energy inside the pitch corridor.

    Returns ``(energy, frame_indices, fps, total_frames)`` where ``energy[i]``
    is the count of corridor pixels that changed noticeably between sampled
    frame ``frame_indices[i]`` and the previous sampled frame (0 for the
    first sample).
    """
    if config is None:
        config = load_config()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1:
        fps = float(config.get("camera", {}).get("fps", 120))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    stride = max(1, int(sample_stride))
    energies: list[float] = []
    frame_indices: list[int] = []
    prev_corridor: np.ndarray | None = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            h, w = gray.shape[:2]
            y0, y1 = int(_CORRIDOR_Y[0] * h), int(_CORRIDOR_Y[1] * h)
            x0, x1 = int(_CORRIDOR_X[0] * w), int(_CORRIDOR_X[1] * w)
            corridor = gray[y0:y1, x0:x1]

            if prev_corridor is None or prev_corridor.shape != corridor.shape:
                energies.append(0.0)
            else:
                diff = cv2.absdiff(corridor, prev_corridor)
                _, moving = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
                energies.append(float(np.count_nonzero(moving)))
            prev_corridor = corridor
            frame_indices.append(frame_idx)
        frame_idx += 1

    cap.release()
    return (
        np.asarray(energies, dtype=np.float64),
        np.asarray(frame_indices, dtype=np.int64),
        fps,
        total,
    )


def _video_fps_and_total(video_path: str | Path, config: dict) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return float(config.get("camera", {}).get("fps", 120)), 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 1:
        fps = float(config.get("camera", {}).get("fps", 120))
    return fps, total


def _default_window(video_path: str | Path, config: dict) -> tuple[int, int]:
    """Fallback window: whole video, or the first ~3s if that's shorter."""
    fps, total = _video_fps_and_total(video_path, config)
    if total <= 1:
        return 0, max(total - 1, 0)
    short_end = min(total - 1, max(1, int(round(3.0 * fps))))
    return 0, short_end


def suggest_flight_window(
    video_path: str | Path,
    *,
    pad_s: float = 0.3,
    sample_stride: int = 2,
    config: dict | None = None,
) -> tuple[int, int]:
    """Suggest a ``(start_frame, end_frame)`` window around the delivery.

    Finds the dominant contiguous burst of motion energy in the pitch
    corridor (the ball flying through + associated arm/release motion),
    then pads it by ``pad_s`` seconds on each side.
    """
    if config is None:
        config = load_config()

    try:
        energies, frame_indices, fps, total = motion_energy_series(
            video_path, sample_stride=sample_stride, config=config
        )
    except RuntimeError:
        return _default_window(video_path, config)

    if total <= 1 or energies.size == 0:
        return _default_window(video_path, config)

    # Short moving average (~50ms) to smooth single-frame flicker.
    smooth_frames = max(1, int(round(0.05 * fps / max(1, sample_stride))))
    kernel = np.ones(smooth_frames, dtype=np.float64) / smooth_frames
    smooth = np.convolve(energies, kernel, mode="same")

    peak_val = float(smooth.max()) if smooth.size else 0.0
    if peak_val < 8.0:
        # Negligible motion anywhere in the corridor.
        return _default_window(video_path, config)

    peak_idx = int(np.argmax(smooth))
    thresh = 0.35 * peak_val

    lo = peak_idx
    while lo > 0 and smooth[lo - 1] > thresh:
        lo -= 1
    hi = peak_idx
    while hi < len(smooth) - 1 and smooth[hi + 1] > thresh:
        hi += 1

    start_frame = int(frame_indices[lo])
    end_frame = int(frame_indices[hi])

    pad_frames = int(round(pad_s * fps))
    start_frame = max(0, start_frame - pad_frames)
    end_frame = min(total - 1, end_frame + pad_frames)

    if end_frame <= start_frame:
        return _default_window(video_path, config)
    return start_frame, end_frame
