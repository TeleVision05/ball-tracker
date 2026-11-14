"""Render a synthetic delivery as a net-session clip, and draw the demo overlay.

The scene is deliberately plain: a dark mat on grass, faded creases, wooden
stumps and a red ball whose size follows its true distance from the camera.
Colours are chosen so the red ball is the brightest saturated object, which is
what the colour (HSV) detector expects of a red ball.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .pitch import (
    BALL_RADIUS_M,
    PITCH_LENGTH_M,
    STUMP_DIAMETER_M,
    STUMP_HEIGHT_M,
    STUMP_WIDTH_M,
)
from .synthetic import CameraModel

# BGR colours.
BACKDROP = (34, 28, 24)
GRASS = (44, 84, 40)
MAT = (74, 98, 76)
CREASE = (104, 112, 108)
STUMP = (62, 96, 122)
BALL = (90, 90, 255)
TRACK_DOT = (0, 220, 255)
FIT_LINE = (255, 200, 60)
TRUE_LINE = (200, 200, 200)
BOUNCE_MARK = (60, 60, 255)
HUD_TEXT = (240, 240, 240)

POPPING_CREASE_M = 1.22
BOWLING_CREASE_HALF_M = 1.32
POPPING_CREASE_HALF_M = 1.83
PITCH_HALF_WIDTH_M = 1.52


class VideoWriteError(RuntimeError):
    """OpenCV could not open a video writer (usually a missing codec)."""


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Open an MPEG-4 writer or raise ``VideoWriteError``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
    if not writer.isOpened():
        raise VideoWriteError(str(path))
    return writer


def _project(camera: CameraModel, points: np.ndarray) -> np.ndarray:
    return camera.project(np.asarray(points, dtype=np.float64).reshape(-1, 3))


def _fill_ground(
    frame: np.ndarray, camera: CameraModel, x_half: float, y0: float, y1: float, color
) -> None:
    corners = np.array(
        [[-x_half, y0, 0.0], [x_half, y0, 0.0], [x_half, y1, 0.0], [-x_half, y1, 0.0]]
    )
    uv = _project(camera, corners)
    if np.all(np.isfinite(uv)):
        cv2.fillPoly(frame, [np.round(uv).astype(np.int32)], color, cv2.LINE_AA)


def _ground_line(frame: np.ndarray, camera: CameraModel, p0, p1, thickness: int = 2) -> None:
    uv = _project(camera, np.array([[p0[0], p0[1], 0.0], [p1[0], p1[1], 0.0]]))
    if np.all(np.isfinite(uv)):
        a, b = np.round(uv).astype(int)
        cv2.line(frame, tuple(a), tuple(b), CREASE, thickness, cv2.LINE_AA)


def _draw_creases(frame: np.ndarray, camera: CameraModel) -> None:
    for stumps_y, toward in ((0.0, 1.0), (PITCH_LENGTH_M, -1.0)):
        popping_y = stumps_y + toward * POPPING_CREASE_M
        _ground_line(
            frame, camera, (-BOWLING_CREASE_HALF_M, stumps_y), (BOWLING_CREASE_HALF_M, stumps_y)
        )
        _ground_line(
            frame, camera, (-POPPING_CREASE_HALF_M, popping_y), (POPPING_CREASE_HALF_M, popping_y)
        )
        for side in (-1.0, 1.0):
            x = side * BOWLING_CREASE_HALF_M
            behind_y = stumps_y - toward * POPPING_CREASE_M
            _ground_line(frame, camera, (x, behind_y), (x, popping_y))


def _draw_stumps(frame: np.ndarray, camera: CameraModel) -> None:
    half = STUMP_DIAMETER_M / 2.0
    spacing = (STUMP_WIDTH_M - STUMP_DIAMETER_M) / 2.0
    for y in (0.0, PITCH_LENGTH_M):
        for x in (-spacing, 0.0, spacing):
            quad = np.array(
                [
                    [x - half, y, 0.0],
                    [x + half, y, 0.0],
                    [x + half, y, STUMP_HEIGHT_M],
                    [x - half, y, STUMP_HEIGHT_M],
                ]
            )
            uv = _project(camera, quad)
            if np.all(np.isfinite(uv)):
                cv2.fillPoly(frame, [np.round(uv).astype(np.int32)], STUMP, cv2.LINE_AA)


def render_background(camera: CameraModel) -> np.ndarray:
    """Static scene: backdrop, grass, mat, creases and both sets of stumps."""
    w, h = camera.resolution
    frame = np.full((h, w, 3), BACKDROP, dtype=np.uint8)
    _fill_ground(frame, camera, 60.0, -3.5, 400.0, GRASS)
    _fill_ground(frame, camera, PITCH_HALF_WIDTH_M, -POPPING_CREASE_M, PITCH_LENGTH_M + 2.5, MAT)
    _draw_creases(frame, camera)
    _draw_stumps(frame, camera)
    return frame


def ball_radius_px(camera: CameraModel, position_m: np.ndarray) -> float:
    """Apparent ball radius from its true depth along the optical axis."""
    rotation, translation = camera.extrinsic_matrix()
    depth = float((rotation @ position_m + translation)[2])
    focal = float(camera.intrinsic_matrix()[0, 0])
    return BALL_RADIUS_M * focal / max(depth, 1e-3)


def draw_ball(frame: np.ndarray, camera: CameraModel, position_m: np.ndarray) -> None:
    """Draw the ball as an anti-aliased red disc at sub-pixel precision."""
    uv = _project(camera, position_m)[0]
    if not np.all(np.isfinite(uv)):
        return
    shift = 4
    scale = 1 << shift
    centre = (int(round(uv[0] * scale)), int(round(uv[1] * scale)))
    radius = int(round(ball_radius_px(camera, position_m) * scale))
    cv2.circle(frame, centre, max(radius, scale), BALL, -1, cv2.LINE_AA, shift)


def _polyline(frame: np.ndarray, uv: np.ndarray, color, thickness: int) -> None:
    pts = uv[np.all(np.isfinite(uv), axis=1)]
    if len(pts) >= 2:
        cv2.polylines(frame, [np.round(pts).astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    detections_uv: np.ndarray,
    fit_uv: np.ndarray,
    bounce_uv: tuple[float, float] | None,
    hud_lines: list[str],
) -> np.ndarray:
    """Draw detections so far, the fitted path so far, the bounce mark and the HUD."""
    out = frame.copy()
    _polyline(out, fit_uv, FIT_LINE, 3)
    for u, v in detections_uv:
        cv2.circle(out, (int(round(u)), int(round(v))), 3, TRACK_DOT, -1, cv2.LINE_AA)
    if bounce_uv is not None:
        centre = (int(round(bounce_uv[0])), int(round(bounce_uv[1])))
        cv2.drawMarker(out, centre, BOUNCE_MARK, cv2.MARKER_TILTED_CROSS, 26, 3, cv2.LINE_AA)
    draw_hud(out, hud_lines)
    return out


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    """Translucent text panel in the top-left corner."""
    if not lines:
        return
    h = frame.shape[0]
    scale = h / 720.0
    font_scale = 0.62 * scale
    thickness = max(1, int(round(1.4 * scale)))
    line_h = int(34 * scale)
    pad = int(14 * scale)
    text_w = max(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        for line in lines
    )
    width = min(text_w + 2 * pad, frame.shape[1] - 2 * pad)
    height = line_h * len(lines) + pad
    panel = frame[pad : pad + height, pad : pad + width]
    panel[:] = (panel * 0.35).astype(np.uint8)
    for i, text in enumerate(lines):
        y = pad + line_h * (i + 1) - int(8 * scale)
        cv2.putText(
            frame,
            text,
            (2 * pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            HUD_TEXT,
            thickness,
            cv2.LINE_AA,
        )


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def video_codec(path: Path) -> str | None:
    """Codec name of the first video stream, or ``None`` if ffprobe is unavailable."""
    if shutil.which("ffprobe") is None:
        return None
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
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    return probe.stdout.strip() or None


def render_delivery_clip(
    camera: CameraModel,
    positions_at_frames: np.ndarray,
    fps: float,
    out_path: Path,
    preroll_frames: int = 0,
    postroll_frames: int = 0,
) -> int:
    """Write the clip: still pre-roll, one frame per ball position, empty post-roll.

    ``positions_at_frames`` holds the ball centre for each in-flight frame, in
    order. Returns the number of frames written.
    """
    background = render_background(camera)
    writer = open_writer(out_path, fps, camera.resolution)
    written = 0
    try:
        for _ in range(preroll_frames):
            writer.write(background)
            written += 1
        for position in positions_at_frames:
            frame = background.copy()
            draw_ball(frame, camera, position)
            writer.write(frame)
            written += 1
        for _ in range(postroll_frames):
            writer.write(background)
            written += 1
    finally:
        writer.release()
    return written
