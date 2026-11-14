"""Tracks built from manual ball clicks (the Streamlit marking workflow).

Import these through :mod:`balltracker.tracking`.
"""

from __future__ import annotations

from .tracking_core import TrackPoint


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

    Clicked frames are measured (``predicted=False``). Between two clicks, up
    to ``max_gap`` frames after the earlier click are filled when
    ``fill_gaps``, on the straight line between the two clicks. Every gap here
    has a click on both sides, so the fill never extrapolates. (A Kalman filter
    used to do this; its velocity estimate lagged so badly on a real clip that
    filled frames landed behind the last click.)
    """
    if not clicks:
        return []

    dt = 1.0 / max(fps, 1e-6)
    frames = sorted(int(f) for f in clicks)
    track: list[TrackPoint] = [_click_point(frames[0], clicks[frames[0]], dt, radius_px)]
    for prev, nxt in zip(frames, frames[1:]):
        if fill_gaps:
            for f in range(prev + 1, min(nxt, prev + max_gap + 1)):
                track.append(
                    _filled_point(f, (prev, clicks[prev]), (nxt, clicks[nxt]), dt, radius_px)
                )
        track.append(_click_point(nxt, clicks[nxt], dt, radius_px))
    return track


def _filled_point(
    frame_idx: int,
    before: tuple[int, tuple[float, float]],
    after: tuple[int, tuple[float, float]],
    dt: float,
    radius_px: float,
) -> TrackPoint:
    """Linear interpolation between the clicks either side of ``frame_idx``."""
    (f0, (u0, v0)), (f1, (u1, v1)) = before, after
    w = (frame_idx - f0) / float(f1 - f0)
    return TrackPoint(
        frame_idx=frame_idx,
        t_s=frame_idx * dt,
        u=float(u0) + w * (float(u1) - float(u0)),
        v=float(v0) + w * (float(v1) - float(v0)),
        radius_px=radius_px,
        confidence=0.35,
        predicted=True,
    )
