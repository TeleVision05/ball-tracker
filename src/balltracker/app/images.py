"""Annotated pictures for the session app: stump points and ball marks.

Pure image drawing (OpenCV and PIL, no Streamlit). Each function returns the
picture to show and the display scale (display pixels / original pixels).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..landmarks import LANDMARK_PLAIN_LABELS, next_landmark_name

WHITE = (255, 255, 255)
PREDICTION = (255, 80, 80)


def scaled_copy(frame_bgr: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    """A copy no wider than ``max_width`` and its scale (never enlarged)."""
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_width / float(w))
    if scale < 1.0:
        disp = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        disp = frame_bgr.copy()
        scale = 1.0
    return disp, scale


def _put_label(disp: np.ndarray, text: str) -> None:
    cv2.putText(disp, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)


def _to_pil(disp: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))


def setup_prompt(points: dict) -> str:
    """The instruction drawn on the setup picture."""
    next_name = next_landmark_name(points)
    if next_name:
        return f"Click: {LANDMARK_PLAIN_LABELS.get(next_name, next_name)}"
    return "All points placed — save when ready"


def annotated_display_image(
    frame_bgr: np.ndarray,
    points: dict,
    max_width: int = 960,
) -> tuple[Image.Image, float]:
    """Return RGB PIL image with markers and the display scale (display/original)."""
    disp, scale = scaled_copy(frame_bgr, max_width)
    for i, (_name, uv) in enumerate(points.items()):
        u = int(float(uv[0]) * scale)
        v = int(float(uv[1]) * scale)
        cv2.drawMarker(disp, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(
            disp,
            str(i + 1),
            (u + 8, max(16, v - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    _put_label(disp, setup_prompt(points))
    return _to_pil(disp), scale


def _draw_click_trail(
    disp: np.ndarray, clicks: dict, current_frame: int, trail_frames: int, scale: float
) -> None:
    """Marks within ``trail_frames`` of the current frame, joined by a line."""
    nearby = sorted(f for f in clicks if abs(int(f) - current_frame) <= trail_frames)
    pts = [(int(float(clicks[f][0]) * scale), int(float(clicks[f][1]) * scale)) for f in nearby]
    if len(pts) >= 2:
        cv2.polylines(disp, [np.array(pts, dtype=np.int32)], False, (0, 200, 255), 2, cv2.LINE_AA)
    for f, (u, v) in zip(nearby, pts):
        color = (0, 255, 0) if int(f) == current_frame else (0, 180, 255)
        cv2.drawMarker(disp, (u, v), color, cv2.MARKER_CROSS, 14, 2)


def _draw_prediction(disp: np.ndarray, prediction: tuple[float, float], scale: float) -> None:
    pu = int(float(prediction[0]) * scale)
    pv = int(float(prediction[1]) * scale)
    cv2.circle(disp, (pu, pv), 18, PREDICTION, 2, cv2.LINE_AA)
    cv2.drawMarker(disp, (pu, pv), PREDICTION, cv2.MARKER_TILTED_CROSS, 12, 2)
    cv2.putText(
        disp,
        "predicted",
        (pu + 12, max(16, pv - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        PREDICTION,
        1,
        cv2.LINE_AA,
    )


def annotated_mark_image(
    frame_bgr: np.ndarray,
    clicks: dict,
    current_frame: int,
    prediction: tuple[float, float] | None = None,
    trail_frames: int = 12,
    max_width: int = 960,
) -> tuple[Image.Image, float]:
    """Frame with prior click trail, current mark, and optional prediction guide."""
    disp, scale = scaled_copy(frame_bgr, max_width)
    _draw_click_trail(disp, clicks, current_frame, trail_frames, scale)
    if prediction is not None and current_frame not in clicks:
        _draw_prediction(disp, prediction, scale)
    _put_label(disp, f"Frame {current_frame} — click the ball")
    return _to_pil(disp), scale
