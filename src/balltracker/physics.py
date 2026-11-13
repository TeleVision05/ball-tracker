"""3D ballistic trajectory fitting from 2D tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .calibration import CameraCalibration, backproject_ray, project_points
from .pitch import BALL_RADIUS_M, PITCH_LENGTH_M
from .tracking import TrackPoint


@dataclass
class FlightSegment:
    """Ballistic segment with optional linear drag."""

    t0: float
    x0: np.ndarray  # (3,)
    v0: np.ndarray  # (3,)
    drag_coeff: float = 0.0


@dataclass
class TrajectoryFit:
    """Piecewise pre/post-bounce trajectory fit."""

    pre: FlightSegment
    post: FlightSegment | None
    bounce_time_s: float | None
    bounce_pos_m: np.ndarray | None
    times: np.ndarray
    positions_3d: np.ndarray
    reprojection_rmse: float


def propagate(segment: FlightSegment, times_relative: np.ndarray, g: float = 9.81) -> np.ndarray:
    """Propagate segment under gravity and optional linear drag ``-c*v``."""
    t = np.asarray(times_relative, dtype=np.float64).reshape(-1)
    x0 = np.asarray(segment.x0, dtype=np.float64)
    v0 = np.asarray(segment.v0, dtype=np.float64)
    c = float(segment.drag_coeff)
    positions = np.zeros((len(t), 3), dtype=np.float64)

    for i, ti in enumerate(t):
        if c <= 0.0:
            positions[i] = x0 + v0 * ti + 0.5 * np.array([0.0, 0.0, -g]) * ti * ti
        else:
            exp_ct = np.exp(-c * ti)
            positions[i, 0] = x0[0] + v0[0] * (1.0 - exp_ct) / c
            positions[i, 1] = x0[1] + v0[1] * (1.0 - exp_ct) / c
            positions[i, 2] = (
                x0[2] + v0[2] * (1.0 - exp_ct) / c - g * (ti / c - (1.0 - exp_ct) / (c * c))
            )
    return positions


def velocity_at(segment: FlightSegment, t_rel: float, g: float = 9.81) -> np.ndarray:
    """Velocity at time ``t_rel`` after segment start (no drag)."""
    return segment.v0 + np.array([0.0, 0.0, -g], dtype=np.float64) * t_rel


def ray_plane_intersection(
    calib: CameraCalibration,
    uv: tuple[float, float] | np.ndarray,
    plane_z: float = BALL_RADIUS_M,
) -> np.ndarray | None:
    """Intersect camera ray with horizontal plane ``z = plane_z``."""
    origin, direction = backproject_ray(calib, uv)
    if abs(direction[2]) < 1e-9:
        return None
    scale = (plane_z - origin[2]) / direction[2]
    if scale <= 0.0:
        return None
    return origin + scale * direction


def detect_bounce_index(track: list[TrackPoint], calib: CameraCalibration | None = None) -> int:
    """Guess bounce index for a behind-stumps camera.

    Image ``v`` increases downward. The bounce is near the local maximum
    of ``v`` (lowest in the frame) in the latter part of the track. When
    calibration is available, prefer the on-pitch candidate with the largest ``v``.
    """
    if len(track) < 5:
        return max(0, len(track) // 2)

    vs = np.array([tp.v for tp in track], dtype=np.float64)
    start = max(1, len(track) // 3)
    end = len(track) - 1
    local = vs[start:end]
    bounce_idx = int(np.argmax(local) + start)

    if calib is None:
        return bounce_idx

    best_idx = bounce_idx
    best_score = float("inf")
    for idx in range(start, end):
        hit = ray_plane_intersection(calib, (track[idx].u, track[idx].v))
        if hit is None:
            continue
        on_pitch = 2.0 <= hit[1] <= PITCH_LENGTH_M + 1.0 and abs(hit[0]) < 2.5
        if not on_pitch:
            continue
        # Prefer large image-v (low in the frame) and an on-pitch plane hit.
        score = -float(vs[idx]) + 5.0 * abs(hit[0])
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _initial_guess(
    track: list[TrackPoint],
    calib: CameraCalibration,
    bounce_idx: int,
    g: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Release/bounce state from ray–plane intersections (scale-aware)."""
    release_uv = (track[0].u, track[0].v)
    bounce_uv = (track[bounce_idx].u, track[bounce_idx].v)

    release_pos = ray_plane_intersection(calib, release_uv, plane_z=2.0)
    if release_pos is None or not (0.0 <= release_pos[1] <= 6.0):
        release_pos = np.array([0.0, 1.2, 2.0], dtype=np.float64)

    bounce_pos = ray_plane_intersection(calib, bounce_uv, plane_z=BALL_RADIUS_M)
    if bounce_pos is None or not (3.0 <= bounce_pos[1] <= PITCH_LENGTH_M + 2.0):
        bounce_pos = np.array([0.0, 14.0, BALL_RADIUS_M], dtype=np.float64)
    else:
        bounce_pos = bounce_pos.copy()
        bounce_pos[2] = BALL_RADIUS_M

    t0 = track[0].t_s
    t_b = track[bounce_idx].t_s
    dt = max(t_b - t0, 1e-3)
    # x_b = x_r + v0*dt + 0.5*a*dt^2  =>  v0 = (x_b - x_r)/dt - 0.5*a*dt
    accel = np.array([0.0, 0.0, -g], dtype=np.float64)
    v0 = (bounce_pos - release_pos) / dt - 0.5 * accel * dt
    return release_pos, v0, bounce_pos


def _ray_point_distance(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    """Perpendicular distance from a 3D point to a ray."""
    w = point - origin
    return float(np.linalg.norm(w - np.dot(w, direction) * direction))


def _fit_segment(
    track_slice: list[TrackPoint],
    calib: CameraCalibration,
    t0: float,
    x0_init: np.ndarray,
    v0_init: np.ndarray,
    g: float,
    use_drag: bool,
    z_end: float | None = None,
    t_end: float | None = None,
    fix_start: bool = False,
) -> tuple[FlightSegment, np.ndarray, np.ndarray]:
    """Fit one ballistic segment via reprojection + ray-distance residuals."""
    times = np.array([tp.t_s for tp in track_slice], dtype=np.float64)
    uvs = np.array([[tp.u, tp.v] for tp in track_slice], dtype=np.float64)
    t_rel = times - t0
    origins = []
    directions = []
    for uv in uvs:
        o, d = backproject_ray(calib, uv)
        origins.append(o)
        directions.append(d)
    origins_a = np.asarray(origins)
    dirs_a = np.asarray(directions)

    if fix_start:
        # Optimise velocity (and optional drag) only; keep x0 fixed.
        def pack(v0: np.ndarray, drag: float) -> np.ndarray:
            return np.concatenate([v0, [drag]]) if use_drag else v0.copy()

        def unpack(params: np.ndarray) -> tuple[np.ndarray, float]:
            if use_drag:
                return params[:3], float(params[3])
            return params[:3], 0.0

        lower = np.array([-25.0, 8.0, -40.0] + ([0.0] if use_drag else []), dtype=np.float64)
        upper = np.array([25.0, 55.0, 15.0] + ([1.0] if use_drag else []), dtype=np.float64)
        p0 = pack(v0_init, 0.0)
        x0_fixed = x0_init.copy()
    else:

        def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            x0 = params[:3]
            v0 = params[3:6]
            drag = float(params[6]) if use_drag else 0.0
            return x0, v0, drag

        lower = np.array(
            [-3.0, 0.0, 0.8, -25.0, 8.0, -40.0] + ([0.0] if use_drag else []),
            dtype=np.float64,
        )
        upper = np.array(
            [3.0, 6.0, 3.2, 25.0, 55.0, 15.0] + ([1.0] if use_drag else []),
            dtype=np.float64,
        )
        p0 = np.concatenate([x0_init, v0_init] + ([0.0] if use_drag else []))
        x0_fixed = None

    p0 = np.clip(p0, lower[: len(p0)], upper[: len(p0)])

    def residuals(params: np.ndarray) -> np.ndarray:
        if fix_start:
            v0, drag = unpack(params)
            x0 = x0_fixed
        else:
            x0, v0, drag = unpack(params)
        seg = FlightSegment(t0=t0, x0=x0, v0=v0, drag_coeff=drag)
        pos = propagate(seg, t_rel, g=g)
        proj = project_points(calib, pos)
        valid = np.isfinite(proj).all(axis=1)
        reproj = (proj[valid] - uvs[valid]).ravel()
        ray_res = np.array(
            [20.0 * _ray_point_distance(pos[i], origins_a[i], dirs_a[i]) for i in range(len(pos))],
            dtype=np.float64,
        )
        extra: list[float] = []
        if z_end is not None and t_end is not None:
            pos_end = propagate(seg, np.array([t_end - t0]), g=g)[0]
            extra.append(40.0 * (pos_end[2] - z_end))
            # Soft on-pitch bounce.
            extra.append(5.0 * max(0.0, pos_end[1] - (PITCH_LENGTH_M + 1.5)))
            extra.append(5.0 * max(0.0, 2.0 - pos_end[1]))
        return np.concatenate([reproj, ray_res, np.asarray(extra, dtype=np.float64)])

    result = least_squares(
        residuals,
        p0,
        bounds=(lower[: len(p0)], upper[: len(p0)]),
        max_nfev=800,
        ftol=1e-8,
        xtol=1e-8,
    )

    if fix_start:
        v0, drag = unpack(result.x)
        x0 = x0_fixed
    else:
        x0, v0, drag = unpack(result.x)

    seg = FlightSegment(t0=t0, x0=x0, v0=v0, drag_coeff=drag)
    positions = propagate(seg, t_rel, g=g)
    return seg, times, positions


def fit_trajectory(
    track: list[TrackPoint],
    calib: CameraCalibration,
    fps: float,
    g: float = 9.81,
    use_drag: bool = False,
) -> TrajectoryFit:
    """Fit piecewise pre/post-bounce 3D trajectory (gravity + optional drag)."""
    del fps  # times already absolute from TrackPoint.t_s
    if len(track) < 5:
        raise ValueError("Need at least 5 track points for trajectory fit")

    bounce_idx = detect_bounce_index(track, calib=calib)
    bounce_idx = int(np.clip(bounce_idx, 2, len(track) - 2))

    release_init, v0_init, bounce_init = _initial_guess(track, calib, bounce_idx, g)
    pre_slice = track[: bounce_idx + 1]
    t0 = pre_slice[0].t_s
    t_bounce = track[bounce_idx].t_s

    # Seed release on ray at observed release pixel with free depth via optimisation.
    pre_seg, pre_times, pre_pos = _fit_segment(
        pre_slice,
        calib,
        t0=t0,
        x0_init=release_init,
        v0_init=v0_init,
        g=g,
        use_drag=use_drag,
        z_end=BALL_RADIUS_M,
        t_end=t_bounce,
        fix_start=False,
    )

    bounce_pos = propagate(pre_seg, np.array([t_bounce - t0]), g=g)[0]
    # Snap bounce onto pitch plane while keeping XY from the fit / plane cue.
    plane_hit = ray_plane_intersection(calib, (track[bounce_idx].u, track[bounce_idx].v))
    if plane_hit is not None and 3.0 <= plane_hit[1] <= PITCH_LENGTH_M + 1.5:
        bounce_pos = plane_hit.copy()
        bounce_pos[2] = BALL_RADIUS_M
        # Re-fit velocity with fixed end constraint by adjusting v0 from release->bounce.
        dt = max(t_bounce - t0, 1e-3)
        accel = np.array([0.0, 0.0, -g], dtype=np.float64)
        v0_refined = (bounce_pos - pre_seg.x0) / dt - 0.5 * accel * dt
        pre_seg = FlightSegment(
            t0=pre_seg.t0,
            x0=pre_seg.x0,
            v0=v0_refined,
            drag_coeff=pre_seg.drag_coeff,
        )
        pre_pos = propagate(pre_seg, pre_times - t0, g=g)
        bounce_pos = propagate(pre_seg, np.array([t_bounce - t0]), g=g)[0]
        bounce_pos[2] = BALL_RADIUS_M

    bounce_vel = velocity_at(pre_seg, t_bounce - t0, g=g)

    post_seg: FlightSegment | None = None
    post_times = np.array([], dtype=np.float64)
    post_pos = np.zeros((0, 3), dtype=np.float64)

    if bounce_idx < len(track) - 3:
        post_slice = track[bounce_idx:]
        v_post_init = bounce_vel.copy()
        v_post_init[2] = abs(0.75 * bounce_vel[2])
        post_seg, post_times, post_pos = _fit_segment(
            post_slice,
            calib,
            t0=t_bounce,
            x0_init=bounce_pos,
            v0_init=v_post_init,
            g=g,
            use_drag=False,
            fix_start=True,
        )

    if len(post_times):
        all_times = np.concatenate([pre_times, post_times[1:]])
        all_pos = np.concatenate([pre_pos, post_pos[1:]])
    else:
        all_times = pre_times
        all_pos = pre_pos

    n = min(len(track), len(all_times))
    uvs = np.array([[tp.u, tp.v] for tp in track[:n]], dtype=np.float64)
    proj = project_points(calib, all_pos[:n])
    valid = np.isfinite(proj).all(axis=1)
    if np.any(valid):
        err = proj[valid] - uvs[valid]
        rmse = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    else:
        rmse = float("inf")

    return TrajectoryFit(
        pre=pre_seg,
        post=post_seg,
        bounce_time_s=t_bounce,
        bounce_pos_m=bounce_pos,
        times=all_times,
        positions_3d=all_pos,
        reprojection_rmse=rmse,
    )
