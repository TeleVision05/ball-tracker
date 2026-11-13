"""Track points, the Kalman filter, detection association and the ballistic filter.

Import these through :mod:`balltracker.tracking`; this module exists so the
tracker stays in small, focused files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import BallDetection


@dataclass
class TrackPoint:
    """One point on the 2D ball track."""

    frame_idx: int
    t_s: float
    u: float
    v: float
    radius_px: float
    confidence: float
    predicted: bool


class _Kalman2D:
    """Constant-velocity Kalman filter for (u, v)."""

    def __init__(self, dt: float) -> None:
        self.dt = dt
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4) * 100.0
        self.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.Q = np.diag([1.0, 1.0, 20.0, 20.0]) * 0.15
        self.R = np.eye(2) * 9.0
        self.initialised = False

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z: np.ndarray) -> None:
        if not self.initialised:
            self.x[:2] = z
            self.initialised = True
            return
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


def _predicted_point(
    frame_idx: int, t_s: float, pred: np.ndarray, radius_px: float, confidence: float
) -> TrackPoint:
    return TrackPoint(
        frame_idx=frame_idx,
        t_s=t_s,
        u=float(pred[0]),
        v=float(pred[1]),
        radius_px=float(radius_px),
        confidence=confidence,
        predicted=True,
    )


def _measured_point(det: BallDetection, frame_idx: int, t_s: float) -> TrackPoint:
    return TrackPoint(
        frame_idx=frame_idx,
        t_s=t_s,
        u=float(det.u),
        v=float(det.v),
        radius_px=float(det.radius_px),
        confidence=float(det.confidence),
        predicted=False,
    )


def associate_detections(
    detections: list[BallDetection | None],
    fps: float,
    max_gap: int = 4,
    max_jump_px: float = 55.0,
    early_max_jump_px: float | None = None,
    early_count: int = 8,
) -> list[TrackPoint]:
    """Associate detections into one smooth track.

    Outlier detections (large jump vs prediction) are **rejected**, not used
    to start a new track — that was causing spaghetti overlays in nets footage.

    ``early_max_jump_px`` (if set) relaxes the jump gate for the first
    ``early_count`` measured points, since a track that has just picked up
    the ball near release/a hand-seeded gate can have a larger initial
    Kalman residual before velocity locks in.
    """
    dt = 1.0 / max(fps, 1e-6)
    kf = _Kalman2D(dt)
    track: list[TrackPoint] = []
    gap = 0
    measured_count = 0

    for frame_idx, det in enumerate(detections):
        t_s = frame_idx * dt
        if det is not None:
            z = np.array([det.u, det.v], dtype=np.float64)
            if kf.initialised:
                pred = kf.predict()
                early = early_max_jump_px is not None and measured_count < early_count
                jump_limit = early_max_jump_px if early else max_jump_px
                if np.linalg.norm(pred - z) > jump_limit:
                    # Reject outlier; optionally predict through the gap.
                    if gap < max_gap:
                        gap += 1
                        radius = track[-1].radius_px if track else det.radius_px
                        track.append(_predicted_point(frame_idx, t_s, pred, radius, 0.2))
                    continue
            kf.update(z)
            track.append(_measured_point(det, frame_idx, t_s))
            gap = 0
            measured_count += 1
        elif kf.initialised and gap < max_gap:
            pred = kf.predict()
            gap += 1
            radius = track[-1].radius_px if track else 6.0
            track.append(_predicted_point(frame_idx, t_s, pred, radius, 0.25))
        # Otherwise lock is lost: stop extending and keep the existing track.
    return track


def _split_segments(measured: list[TrackPoint], min_points: int) -> list[list[TrackPoint]]:
    """Split measured points on large jumps or frame gaps; keep long runs."""
    segments: list[list[TrackPoint]] = []
    current: list[TrackPoint] = [measured[0]]
    for prev, nxt in zip(measured, measured[1:]):
        jump = float(np.hypot(nxt.u - prev.u, nxt.v - prev.v))
        frame_gap = nxt.frame_idx - prev.frame_idx
        if jump > 70.0 or frame_gap > 5:
            if len(current) >= min_points:
                segments.append(current)
            current = [nxt]
        else:
            current.append(nxt)
    if len(current) >= min_points:
        segments.append(current)
    return segments


def _segment_score(seg: list[TrackPoint]) -> float:
    """Prefer longer tracks with consistent non-trivial motion (ball flight)."""
    if len(seg) < 2:
        return 0.0
    dists = [
        float(np.hypot(seg[i].u - seg[i - 1].u, seg[i].v - seg[i - 1].v))
        for i in range(1, len(seg))
    ]
    mean_step = float(np.mean(dists))
    if mean_step < 2.0:
        return 0.0
    return len(seg) * mean_step


def filter_ballistic_segment(track: list[TrackPoint], min_points: int = 6) -> list[TrackPoint]:
    """Keep the longest nearly-smooth run of measured (non-predicted) points."""
    if len(track) < min_points:
        return track

    measured = [tp for tp in track if not tp.predicted]
    if len(measured) < min_points:
        return track

    segments = _split_segments(measured, min_points)
    if not segments:
        return measured

    best = max(segments, key=_segment_score)
    # Fit a quadratic in time to reject residual outliers.
    frames = np.array([tp.frame_idx for tp in best], dtype=np.float64)
    us = np.array([tp.u for tp in best], dtype=np.float64)
    vs = np.array([tp.v for tp in best], dtype=np.float64)
    if len(best) >= 8:
        cu = np.polyfit(frames, us, 2)
        cv = np.polyfit(frames, vs, 2)
        uref = np.polyval(cu, frames)
        vref = np.polyval(cv, frames)
        resid = np.hypot(us - uref, vs - vref)
        keep = resid < max(12.0, 2.5 * np.median(resid))
        filtered = [tp for tp, k in zip(best, keep) if k]
        if len(filtered) >= min_points:
            return filtered
    return best
