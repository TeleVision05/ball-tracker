"""Tracks built from manual ball clicks (the Streamlit marking workflow).

Import these through :mod:`balltracker.tracking`.
"""

from __future__ import annotations

import numpy as np

from .tracking_core import TrackPoint, _Kalman2D


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


def _click_point(
    frame_idx: int, uv: tuple[float, float], dt: float, radius_px: float
) -> TrackPoint:
    return TrackPoint(
        frame_idx=frame_idx,
        t_s=frame_idx * dt,
        u=float(uv[0]),
        v=float(uv[1]),
        radius_px=radius_px,
        confidence=1.0,
        predicted=False,
    )


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
        return [_click_point(f, clicks[f], dt, radius_px) for f in frames]

    kf = _Kalman2D(dt)
    first = frames[0]
    u0, v0 = clicks[first]
    kf.update(np.array([u0, v0], dtype=np.float64))
    track: list[TrackPoint] = [_click_point(first, (u0, v0), dt, radius_px)]

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
                track.append(_click_point(nxt, (u, v), dt, radius_px))
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
