"""2D ball tracking across video frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .errors import VideoOpenError
from .config import load_config
from .detection import (
    BallDetection,
    _score_motion_candidates,
    detect_ball_hsv,
    detect_ball_white,
    detect_video,
)
from .release import ReleaseSeed, estimate_release_seed


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
                jump_limit = (
                    early_max_jump_px
                    if early_max_jump_px is not None and measured_count < early_count
                    else max_jump_px
                )
                if np.linalg.norm(pred - z) > jump_limit:
                    # Reject outlier; optionally predict through the gap.
                    if gap < max_gap:
                        gap += 1
                        track.append(
                            TrackPoint(
                                frame_idx=frame_idx,
                                t_s=t_s,
                                u=float(pred[0]),
                                v=float(pred[1]),
                                radius_px=float(track[-1].radius_px if track else det.radius_px),
                                confidence=0.2,
                                predicted=True,
                            )
                        )
                    continue
                kf.update(z)
            else:
                kf.update(z)
            track.append(
                TrackPoint(
                    frame_idx=frame_idx,
                    t_s=t_s,
                    u=float(det.u),
                    v=float(det.v),
                    radius_px=float(det.radius_px),
                    confidence=float(det.confidence),
                    predicted=False,
                )
            )
            gap = 0
            measured_count += 1
        elif kf.initialised and gap < max_gap:
            pred = kf.predict()
            gap += 1
            track.append(
                TrackPoint(
                    frame_idx=frame_idx,
                    t_s=t_s,
                    u=float(pred[0]),
                    v=float(pred[1]),
                    radius_px=float(track[-1].radius_px if track else 6.0),
                    confidence=0.25,
                    predicted=True,
                )
            )
        elif kf.initialised and gap >= max_gap:
            # Lost lock — stop extending; keep existing track only.
            pass
    return track


def filter_ballistic_segment(track: list[TrackPoint], min_points: int = 6) -> list[TrackPoint]:
    """Keep the longest nearly-smooth run of measured (non-predicted) points."""
    if len(track) < min_points:
        return track

    measured = [tp for tp in track if not tp.predicted]
    if len(measured) < min_points:
        return track

    # Split on large jumps / frame gaps.
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

    if not segments:
        return measured

    def _score(seg: list[TrackPoint]) -> float:
        if len(seg) < 2:
            return 0.0
        dists = [
            float(np.hypot(seg[i].u - seg[i - 1].u, seg[i].v - seg[i - 1].v))
            for i in range(1, len(seg))
        ]
        # Prefer longer tracks with consistent non-trivial motion (ball flight).
        mean_step = float(np.mean(dists))
        if mean_step < 2.0:
            return 0.0
        return len(seg) * mean_step

    best = max(segments, key=_score)
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

    if end_frame is None or end_frame < 0:
        end_frame = max(0, total - 1)
    start_frame = int(max(0, start_frame))
    end_frame = int(min(end_frame, max(0, total - 1)))
    if end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame

    if release_seed is None and det_cfg.get("use_pose_release", True):
        try:
            release_seed = estimate_release_seed(
                video_path, start_frame=start_frame, end_frame=end_frame, config=config
            )
        except Exception:
            release_seed = None

    detection_start = start_frame
    gate_uv: tuple[float, float] | None = None
    direction_uv: tuple[float, float] | None = None
    if release_seed is not None:
        detection_start = max(start_frame, release_seed.frame_idx - 2)
        gate_uv = (release_seed.u, release_seed.v)
        direction_uv = release_seed.direction_uv

    min_r = float(det_cfg.get("min_ball_radius_px", 2))
    max_r = float(det_cfg.get("max_ball_radius_px", 22))
    min_circularity = float(det_cfg.get("gated_min_circularity", 0.15))
    base_gate_px = float(det_cfg.get("release_max_gate_px", 120.0))
    use_csrt = bool(det_cfg.get("use_csrt", True))

    mog2 = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=16, detectShadows=False)

    # Warm the background model on frames before the detection window (or,
    # if there's no room before start_frame, on the first few frames of it).
    warm_start = max(0, start_frame - 30)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(warm_start))
    warm_frames = (start_frame - warm_start) if warm_start < start_frame else 5
    for _ in range(warm_frames):
        ok, frame = cap.read()
        if not ok:
            break
        mog2.apply(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    detections: list[BallDetection | None] = [None] * total
    csrt = None
    missed_streak = 0
    last_gate: tuple[float, float] | None = gate_uv

    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        fg = mog2.apply(frame)

        if frame_idx < detection_start:
            frame_idx += 1
            continue

        det: BallDetection | None = None
        gate_px = base_gate_px * (1.0 + 0.2 * missed_streak)

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
        gated_fg = cv2.bitwise_and(fg, roi_mask)

        candidates = _score_motion_candidates(
            gated_fg,
            frame.shape,
            min_r,
            max_r,
            gate_uv=gate_uv,
            max_gate_px=gate_px,
            direction_uv=direction_uv,
            prefer_elongated=True,
            min_circularity=min_circularity,
            gated=gate_uv is not None,
        )
        if candidates:
            u, v, radius_px, score = candidates[0]
            conf = float(np.clip(score, 0.05, 0.95))
            det = BallDetection(frame_idx=frame_idx, u=u, v=v, radius_px=radius_px, confidence=conf)
        elif gate_uv is not None:
            # Bounded colour fallback: a small crop around the gate, not full-frame roam.
            crop_r = max(30.0, gate_px * 0.6)
            cx0 = int(max(0, gate_uv[0] - crop_r))
            cx1 = int(min(w, gate_uv[0] + crop_r))
            cy0 = int(max(0, gate_uv[1] - crop_r))
            cy1 = int(min(h, gate_uv[1] + crop_r))
            crop = frame[cy0:cy1, cx0:cx1]
            if crop.size > 0:
                cand = detect_ball_hsv(crop, config) or detect_ball_white(crop, config)
                if cand is not None:
                    det = BallDetection(
                        frame_idx=frame_idx,
                        u=cand.u + cx0,
                        v=cand.v + cy0,
                        radius_px=cand.radius_px,
                        confidence=float(min(0.6, cand.confidence)),
                    )

        if det is None and csrt is not None:
            ok_csrt, bbox = csrt.update(frame)
            if ok_csrt:
                bx, by, bw, bh = bbox
                det = BallDetection(
                    frame_idx=frame_idx,
                    u=float(bx + bw / 2.0),
                    v=float(by + bh / 2.0),
                    radius_px=float(max(bw, bh) / 2.0),
                    confidence=0.3,
                )
            else:
                csrt = None

        if det is not None:
            detections[frame_idx] = det
            if last_gate is not None:
                du, dv = det.u - last_gate[0], det.v - last_gate[1]
                dn = float(np.hypot(du, dv))
                if dn > 1.0:
                    direction_uv = (du / dn, dv / dn)
            last_gate = (det.u, det.v)
            gate_uv = (det.u, det.v)
            missed_streak = 0

            if use_csrt and csrt is None and det.confidence > 0.35:
                tracker = _create_csrt_tracker()
                if tracker is not None:
                    size = max(12.0, det.radius_px * 4.0)
                    bbox = (
                        int(max(0, det.u - size / 2)),
                        int(max(0, det.v - size / 2)),
                        int(size),
                        int(size),
                    )
                    try:
                        tracker.init(frame, bbox)
                        csrt = tracker
                    except Exception:
                        csrt = None
        else:
            missed_streak += 1

        frame_idx += 1

    cap.release()

    track = associate_detections(
        detections,
        fps=fps,
        max_jump_px=float(det_cfg.get("hybrid_max_jump_px", 55.0)),
        early_max_jump_px=float(det_cfg.get("hybrid_early_max_jump_px", 90.0)),
    )

    if release_seed is not None and track and track[0].frame_idx > release_seed.frame_idx:
        seed_point = TrackPoint(
            frame_idx=release_seed.frame_idx,
            t_s=release_seed.frame_idx / max(fps, 1e-6),
            u=release_seed.u,
            v=release_seed.v,
            radius_px=track[0].radius_px,
            confidence=0.5,
            predicted=False,
        )
        track = [seed_point] + track

    return filter_ballistic_segment(track)


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


def predict_ball_uv(
    clicks: dict[int, tuple[float, float]],
    frame_idx: int,
    fps: float,
) -> tuple[float, float] | None:
    """Predict ball ``(u, v)`` at ``frame_idx`` from prior manual clicks.

    Uses constant-velocity extrapolation from the two most recent clicks at or
    before ``frame_idx``. With only one prior click, returns that position.
    ``fps`` is accepted for API symmetry with tracking helpers.
    """
    del fps  # reserved; times are already in frame units here
    if not clicks:
        return None

    prior = sorted(f for f in clicks if f <= frame_idx)
    if not prior:
        earliest = min(clicks)
        u, v = clicks[earliest]
        return float(u), float(v)

    if prior[-1] == frame_idx:
        u, v = clicks[frame_idx]
        return float(u), float(v)

    if len(prior) == 1:
        u, v = clicks[prior[0]]
        return float(u), float(v)

    f0, f1 = prior[-2], prior[-1]
    u0, v0 = clicks[f0]
    u1, v1 = clicks[f1]
    df = float(f1 - f0)
    if abs(df) < 1e-9:
        return float(u1), float(v1)
    du = (float(u1) - float(u0)) / df
    dv = (float(v1) - float(v0)) / df
    steps = float(frame_idx - f1)
    return float(u1) + du * steps, float(v1) + dv * steps


def track_from_clicks(
    clicks: dict[int, tuple[float, float]],
    fps: float,
    fill_gaps: bool = True,
    max_gap: int = 8,
    radius_px: float = 6.0,
) -> list[TrackPoint]:
    """Build a track from manual frame clicks, optionally filling gaps.

    Clicked frames are measured (``predicted=False``). Between clicks, up to
    ``max_gap`` frames are filled with Kalman predictions when ``fill_gaps``.
    """
    if not clicks:
        return []

    dt = 1.0 / max(fps, 1e-6)
    frames = sorted(int(f) for f in clicks)

    if not fill_gaps:
        return [
            TrackPoint(
                frame_idx=f,
                t_s=f * dt,
                u=float(clicks[f][0]),
                v=float(clicks[f][1]),
                radius_px=radius_px,
                confidence=1.0,
                predicted=False,
            )
            for f in frames
        ]

    kf = _Kalman2D(dt)
    track: list[TrackPoint] = []
    first = frames[0]
    u0, v0 = clicks[first]
    kf.update(np.array([u0, v0], dtype=np.float64))
    track.append(
        TrackPoint(
            frame_idx=first,
            t_s=first * dt,
            u=float(u0),
            v=float(v0),
            radius_px=radius_px,
            confidence=1.0,
            predicted=False,
        )
    )

    for nxt in frames[1:]:
        prev = track[-1].frame_idx
        gap = nxt - prev
        if gap <= 0:
            continue

        # One predict per frame from prev → nxt, record near gaps only.
        for step in range(1, gap + 1):
            pred = kf.predict()
            f = prev + step
            if f == nxt:
                u, v = clicks[nxt]
                kf.update(np.array([u, v], dtype=np.float64))
                track.append(
                    TrackPoint(
                        frame_idx=nxt,
                        t_s=nxt * dt,
                        u=float(u),
                        v=float(v),
                        radius_px=radius_px,
                        confidence=1.0,
                        predicted=False,
                    )
                )
            elif step <= max_gap:
                track.append(
                    TrackPoint(
                        frame_idx=f,
                        t_s=f * dt,
                        u=float(pred[0]),
                        v=float(pred[1]),
                        radius_px=radius_px,
                        confidence=0.35,
                        predicted=True,
                    )
                )

    return track
