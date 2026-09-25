"""Ball detection for synthetic (red) and real net footage (white / motion)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import load_config


@dataclass
class BallDetection:
    """Single-frame ball detection."""

    frame_idx: int
    u: float
    v: float
    radius_px: float
    confidence: float


def _red_hsv_ranges(config: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return lower/upper HSV bounds for red (handles hue wrap)."""
    det = config.get("detection", {})
    low = np.array(det.get("ball_color_hsv_low", [0, 120, 120]), dtype=np.uint8)
    high = np.array(det.get("ball_color_hsv_high", [10, 255, 255]), dtype=np.uint8)
    ranges = [(low, high)]
    ranges.append(
        (
            np.array([170, low[1], low[2]], dtype=np.uint8),
            np.array([180, 255, 255], dtype=np.uint8),
        )
    )
    return ranges


def _best_circular_blob(
    mask: np.ndarray,
    min_r: int,
    max_r: int,
    conf_thresh: float,
    frame_shape: tuple[int, int],
) -> BallDetection | None:
    h, w = frame_shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = -1.0
    for cnt in contours:
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < min_r or radius > max_r:
            continue
        # Ignore detections on the border (net posts / players walking).
        if cx < 0.08 * w or cx > 0.92 * w or cy < 0.05 * h or cy > 0.92 * h:
            continue
        area = cv2.contourArea(cnt)
        circle_area = max(np.pi * radius * radius, 1.0)
        circularity = float(area / circle_area)
        if circularity < 0.35:
            continue
        score = circularity * min(1.0, area / 60.0)
        if score > best_score:
            best_score = score
            best = (float(cx), float(cy), float(radius), float(min(1.0, score)))

    if best is None or best[3] < conf_thresh:
        return None
    u, v, radius_px, confidence = best
    return BallDetection(frame_idx=-1, u=u, v=v, radius_px=radius_px, confidence=confidence)


def detect_ball_hsv(frame_bgr: np.ndarray, config: dict | None = None) -> BallDetection | None:
    """Detect red ball in a BGR frame using HSV thresholding."""
    if config is None:
        config = load_config()
    det_cfg = config.get("detection", {})
    min_r = int(det_cfg.get("min_ball_radius_px", 3))
    max_r = int(det_cfg.get("max_ball_radius_px", 28))
    conf_thresh = float(det_cfg.get("confidence_threshold", 0.45))

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in _red_hsv_ranges(config):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
    return _best_circular_blob(mask, min_r, max_r, conf_thresh, frame_bgr.shape)


def detect_ball_white(frame_bgr: np.ndarray, config: dict | None = None) -> BallDetection | None:
    """Detect a bright / white cricket ball (common in training nets)."""
    if config is None:
        config = load_config()
    det_cfg = config.get("detection", {})
    min_r = int(det_cfg.get("min_ball_radius_px", 2))
    max_r = int(det_cfg.get("max_ball_radius_px", 22))
    conf_thresh = float(det_cfg.get("confidence_threshold", 0.35))

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Low saturation, high value = white / bright leather.
    mask = cv2.inRange(hsv, np.array([0, 0, 175], dtype=np.uint8), np.array([180, 70, 255], dtype=np.uint8))
    return _best_circular_blob(mask, min_r, max_r, conf_thresh, frame_bgr.shape)


def refine_center_subpixel(frame: np.ndarray, detection: BallDetection) -> BallDetection:
    """Refine ball centre using image moments inside a local crop (color path)."""
    h, w = frame.shape[:2]
    pad = int(max(8, detection.radius_px * 2))
    u0 = int(max(0, detection.u - pad))
    v0 = int(max(0, detection.v - pad))
    u1 = int(min(w, detection.u + pad))
    v1 = int(min(h, detection.v + pad))
    crop = frame[v0:v1, u0:u1]
    if crop.size == 0:
        return detection

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] > 1e-3:
        cu = m["m10"] / m["m00"] + u0
        cv_ = m["m01"] / m["m00"] + v0
    else:
        cu, cv_ = detection.u, detection.v

    return BallDetection(
        frame_idx=detection.frame_idx,
        u=float(cu),
        v=float(cv_),
        radius_px=detection.radius_px,
        confidence=detection.confidence,
    )


def detect_ball_motion(
    frame_bgr: np.ndarray,
    prev_gray: np.ndarray | None,
    config: dict | None = None,
    gate_uv: tuple[float, float] | None = None,
    max_gate_px: float = 70.0,
) -> tuple[BallDetection | None, np.ndarray]:
    """Detect a small, fast-moving ball-like blob (gated when a prior exists)."""
    if config is None:
        config = load_config()
    det_cfg = config.get("detection", {})
    min_r = int(det_cfg.get("min_ball_radius_px", 2))
    max_r = int(det_cfg.get("max_ball_radius_px", 18))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if prev_gray is None:
        return None, gray

    diff = cv2.absdiff(gray, prev_gray)
    # Higher threshold to ignore net mesh shimmer.
    _, mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Prefer pitch corridor (central horizontal band).
    h, w = frame_bgr.shape[:2]
    roi = np.zeros_like(mask)
    y0, y1 = int(0.15 * h), int(0.85 * h)
    x0, x1 = int(0.1 * w), int(0.9 * w)
    roi[y0:y1, x0:x1] = 255
    mask = cv2.bitwise_and(mask, roi)

    det = _best_circular_blob(mask, min_r, max_r, conf_thresh=0.25, frame_shape=frame_bgr.shape)
    if det is None:
        return None, gray

    if gate_uv is not None:
        dist = float(np.hypot(det.u - gate_uv[0], det.v - gate_uv[1]))
        if dist > max_gate_px:
            return None, gray

    return det, gray


def _score_motion_candidates(
    mask: np.ndarray,
    frame_shape: tuple[int, int],
    min_r: float,
    max_r: float,
    gate_uv: tuple[float, float] | None = None,
    max_gate_px: float = 100.0,
    direction_uv: tuple[float, float] | None = None,
    prefer_elongated: bool = True,
    min_circularity: float = 0.15,
    gated: bool = False,
) -> list[tuple[float, float, float, float]]:
    """Score motion-blob candidates for the blur-tolerant hybrid tracker.

    Unlike ``_best_circular_blob`` (single best-or-None, strict circularity),
    this returns every plausible candidate ``(u, v, radius_px, score)``
    sorted best-first, with a looser circularity floor so motion-blurred
    (elongated) ball blobs aren't discarded. When ``gate_uv`` is given,
    candidates farther than ``max_gate_px`` are rejected outright, and
    candidates near the gate / continuing along ``direction_uv`` (the
    recent travel direction) are boosted. ``gated=True`` loosens the border
    reject, since a bowler near the frame edge shouldn't disqualify a
    close-to-hand release seed.
    """
    h, w = frame_shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    border_x = 0.03 * w if gated else 0.08 * w
    border_y = 0.03 * h if gated else 0.05 * h

    dir_vec: tuple[float, float] | None = None
    if direction_uv is not None:
        dn = float(np.hypot(direction_uv[0], direction_uv[1]))
        if dn > 1e-6:
            dir_vec = (direction_uv[0] / dn, direction_uv[1] / dn)

    circ_floor = min_circularity if prefer_elongated else max(min_circularity, 0.35)

    results: list[tuple[float, float, float, float]] = []
    for cnt in contours:
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < min_r or radius > max_r:
            continue
        if cx < border_x or cx > w - border_x or cy < border_y or cy > h - border_y:
            continue

        area = cv2.contourArea(cnt)
        circle_area = max(np.pi * radius * radius, 1.0)
        circularity = float(area / circle_area)
        if circularity < circ_floor:
            continue

        score = circularity * min(1.0, area / 50.0)

        if gate_uv is not None:
            dist = float(np.hypot(cx - gate_uv[0], cy - gate_uv[1]))
            if dist > max_gate_px:
                continue
            score += 0.6 * (1.0 - dist / max_gate_px)

            if dir_vec is not None and dist > 1e-3:
                to_cand = ((cx - gate_uv[0]) / dist, (cy - gate_uv[1]) / dist)
                alignment = to_cand[0] * dir_vec[0] + to_cand[1] * dir_vec[1]
                # Reward candidates ahead of the gate along the travel direction.
                score += 0.5 * max(0.0, alignment)

        if dir_vec is not None and len(cnt) >= 5:
            try:
                (_, _), (major, minor), angle = cv2.fitEllipse(cnt)
            except cv2.error:
                major = minor = 0.0
            if major > minor * 1.2:
                theta = np.radians(angle)
                axis = (float(np.sin(theta)), float(-np.cos(theta)))
                axis_align = abs(axis[0] * dir_vec[0] + axis[1] * dir_vec[1])
                score += 0.3 * axis_align

        results.append((float(cx), float(cy), float(radius), float(score)))

    results.sort(key=lambda item: item[3], reverse=True)
    return results


def detect_video(
    video_path: str | Path,
    config: dict | None = None,
    refine: bool = True,
    method: str = "auto",
    start_frame: int = 0,
    end_frame: int | None = None,
) -> list[BallDetection | None]:
    """Detect ball on frames ``[start_frame, end_frame]`` (inclusive).

    Returned list is indexed from 0 of the *full* video: entries outside the
    analysis window are ``None``. ``method``: ``auto`` | ``hsv`` | ``white`` |
    ``motion``.
    """
    if config is None:
        config = load_config()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None or end_frame < 0:
        end_frame = max(0, total - 1)
    start_frame = int(max(0, start_frame))
    end_frame = int(min(end_frame, max(0, total - 1)))
    if end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame

    detections: list[BallDetection | None] = [None] * total
    prev_gray: np.ndarray | None = None
    gate_uv: tuple[float, float] | None = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx < start_frame:
            # Keep background model warm a few frames before the window.
            if frame_idx >= start_frame - 2:
                prev_gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            frame_idx += 1
            continue
        if frame_idx > end_frame:
            break

        det: BallDetection | None = None
        used_color = False

        if method in ("hsv", "auto"):
            det = detect_ball_hsv(frame, config)
            used_color = det is not None
        if det is None and method in ("white", "auto"):
            det = detect_ball_white(frame, config)
            used_color = det is not None
        if det is None and method in ("motion", "auto"):
            det, prev_gray = detect_ball_motion(
                frame, prev_gray, config, gate_uv=gate_uv, max_gate_px=80.0
            )
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if det is not None:
            det = BallDetection(
                frame_idx=frame_idx,
                u=det.u,
                v=det.v,
                radius_px=det.radius_px,
                confidence=det.confidence,
            )
            if refine and used_color:
                det = refine_center_subpixel(frame, det)
            # Once we have a lock, gate future motion detections near it.
            gate_uv = (det.u, det.v)
            detections[frame_idx] = det
        elif gate_uv is not None:
            # Slowly forget gate if we lose the ball for many frames.
            pass

        frame_idx += 1

    cap.release()
    return detections
