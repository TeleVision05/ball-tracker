"""Per-frame detection for the hybrid tracker: gated MOG2 motion, colour, CSRT.

:func:`balltracker.tracking.track_delivery_hybrid` opens the clip, picks the
release seed and turns these detections into a track. This module holds the
frame loop it delegates to, split into one step per function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .detection import (
    BallDetection,
    _score_motion_candidates,
    detect_ball_hsv,
    detect_ball_white,
)
from .release import ReleaseSeed

UV = tuple[float, float]


def _create_csrt_tracker():
    """Best-effort CSRT tracker construction across OpenCV build variants."""
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        pass
    try:
        return cv2.legacy.TrackerCSRT_create()
    except Exception:
        return None


@dataclass(frozen=True)
class HybridSettings:
    """The ``detection`` config knobs the frame loop reads."""

    min_r: float
    max_r: float
    min_circularity: float
    base_gate_px: float
    use_csrt: bool

    @classmethod
    def from_config(cls, det_cfg: dict) -> HybridSettings:
        return cls(
            min_r=float(det_cfg.get("min_ball_radius_px", 2)),
            max_r=float(det_cfg.get("max_ball_radius_px", 22)),
            min_circularity=float(det_cfg.get("gated_min_circularity", 0.15)),
            base_gate_px=float(det_cfg.get("release_max_gate_px", 120.0)),
            use_csrt=bool(det_cfg.get("use_csrt", True)),
        )


@dataclass
class _GateState:
    """Where the search gate is, which way the ball is going, and the CSRT lock."""

    gate_uv: UV | None = None
    direction_uv: UV | None = None
    last_gate: UV | None = None
    missed_streak: int = 0
    csrt: Any = None


def _warm_background(cap: cv2.VideoCapture, mog2: Any, start_frame: int) -> None:
    """Warm the background model on frames before the detection window.

    If there is no room before ``start_frame``, warm on the first few frames
    of the window instead. Leaves the capture positioned at ``start_frame``.
    """
    warm_start = max(0, start_frame - 30)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(warm_start))
    warm_frames = (start_frame - warm_start) if warm_start < start_frame else 5
    for _ in range(warm_frames):
        ok, frame = cap.read()
        if not ok:
            break
        mog2.apply(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))


def _gate_mask(fg: np.ndarray, gate_uv: UV | None, gate_px: float) -> np.ndarray:
    """Foreground restricted to the gate square, or the frame's central area."""
    h, w = fg.shape[:2]
    roi_mask = np.zeros_like(fg)
    if gate_uv is not None:
        x0 = int(max(0, gate_uv[0] - gate_px))
        x1 = int(min(w, gate_uv[0] + gate_px))
        y0 = int(max(0, gate_uv[1] - gate_px))
        y1 = int(min(h, gate_uv[1] + gate_px))
        roi_mask[y0:y1, x0:x1] = 255
    else:
        y0, y1 = int(0.15 * h), int(0.85 * h)
        x0, x1 = int(0.10 * w), int(0.90 * w)
        roi_mask[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(fg, roi_mask)


def _motion_detection(
    frame: np.ndarray,
    fg: np.ndarray,
    frame_idx: int,
    state: _GateState,
    settings: HybridSettings,
    gate_px: float,
) -> BallDetection | None:
    """Best moving blob inside the gate, scored for size, shape and direction."""
    candidates = _score_motion_candidates(
        _gate_mask(fg, state.gate_uv, gate_px),
        frame.shape,
        settings.min_r,
        settings.max_r,
        gate_uv=state.gate_uv,
        max_gate_px=gate_px,
        direction_uv=state.direction_uv,
        prefer_elongated=True,
        min_circularity=settings.min_circularity,
        gated=state.gate_uv is not None,
    )
    if not candidates:
        return None
    u, v, radius_px, score = candidates[0]
    conf = float(np.clip(score, 0.05, 0.95))
    return BallDetection(frame_idx=frame_idx, u=u, v=v, radius_px=radius_px, confidence=conf)


def _colour_fallback(
    frame: np.ndarray, frame_idx: int, gate_uv: UV, gate_px: float, config: dict
) -> BallDetection | None:
    """Bounded colour fallback: a small crop around the gate, not full-frame roam."""
    h, w = frame.shape[:2]
    crop_r = max(30.0, gate_px * 0.6)
    cx0 = int(max(0, gate_uv[0] - crop_r))
    cx1 = int(min(w, gate_uv[0] + crop_r))
    cy0 = int(max(0, gate_uv[1] - crop_r))
    cy1 = int(min(h, gate_uv[1] + crop_r))
    crop = frame[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return None
    cand = detect_ball_hsv(crop, config) or detect_ball_white(crop, config)
    if cand is None:
        return None
    return BallDetection(
        frame_idx=frame_idx,
        u=cand.u + cx0,
        v=cand.v + cy0,
        radius_px=cand.radius_px,
        confidence=float(min(0.6, cand.confidence)),
    )


def _csrt_detection(state: _GateState, frame: np.ndarray, frame_idx: int) -> BallDetection | None:
    """Ask the locked CSRT tracker; drop the lock when it loses the ball."""
    ok_csrt, bbox = state.csrt.update(frame)
    if not ok_csrt:
        state.csrt = None
        return None
    bx, by, bw, bh = bbox
    return BallDetection(
        frame_idx=frame_idx,
        u=float(bx + bw / 2.0),
        v=float(by + bh / 2.0),
        radius_px=float(max(bw, bh) / 2.0),
        confidence=0.3,
    )


def _detect_frame(
    frame: np.ndarray,
    fg: np.ndarray,
    frame_idx: int,
    state: _GateState,
    settings: HybridSettings,
    config: dict,
) -> BallDetection | None:
    """Motion first, then colour inside the gate, then CSRT."""
    gate_px = settings.base_gate_px * (1.0 + 0.2 * state.missed_streak)
    det = _motion_detection(frame, fg, frame_idx, state, settings, gate_px)
    if det is None and state.gate_uv is not None:
        det = _colour_fallback(frame, frame_idx, state.gate_uv, gate_px, config)
    if det is None and state.csrt is not None:
        det = _csrt_detection(state, frame, frame_idx)
    return det


def _start_csrt(frame: np.ndarray, det: BallDetection) -> Any:
    """A CSRT tracker locked on ``det``, or None if this OpenCV has none."""
    tracker = _create_csrt_tracker()
    if tracker is None:
        return None
    size = max(12.0, det.radius_px * 4.0)
    bbox = (
        int(max(0, det.u - size / 2)),
        int(max(0, det.v - size / 2)),
        int(size),
        int(size),
    )
    try:
        tracker.init(frame, bbox)
        return tracker
    except Exception:
        return None


def _lock_on(state: _GateState, det: BallDetection, frame: np.ndarray, use_csrt: bool) -> None:
    """Move the gate to ``det``, update the direction, and start CSRT if confident."""
    if state.last_gate is not None:
        du, dv = det.u - state.last_gate[0], det.v - state.last_gate[1]
        dn = float(np.hypot(du, dv))
        if dn > 1.0:
            state.direction_uv = (du / dn, dv / dn)
    state.last_gate = (det.u, det.v)
    state.gate_uv = (det.u, det.v)
    state.missed_streak = 0
    if use_csrt and state.csrt is None and det.confidence > 0.35:
        state.csrt = _start_csrt(frame, det)


def detect_hybrid(
    cap: cv2.VideoCapture,
    total: int,
    start_frame: int,
    end_frame: int,
    release_seed: ReleaseSeed | None,
    config: dict,
) -> list[BallDetection | None]:
    """Run the gated detector over ``start_frame..end_frame`` of an open clip.

    Returns one entry per frame of the clip (``total``); frames outside the
    window or without a detection are None.
    """
    settings = HybridSettings.from_config(config.get("detection", {}))
    state = _GateState()
    detection_start = start_frame
    if release_seed is not None:
        detection_start = max(start_frame, release_seed.frame_idx - 2)
        state.gate_uv = (release_seed.u, release_seed.v)
        state.direction_uv = release_seed.direction_uv
    state.last_gate = state.gate_uv

    mog2 = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=16, detectShadows=False)
    _warm_background(cap, mog2, start_frame)

    detections: list[BallDetection | None] = [None] * total
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        fg = mog2.apply(frame)
        if frame_idx >= detection_start:
            det = _detect_frame(frame, fg, frame_idx, state, settings, config)
            if det is not None:
                detections[frame_idx] = det
                _lock_on(state, det, frame, settings.use_csrt)
            else:
                state.missed_streak += 1
        frame_idx += 1
    return detections
