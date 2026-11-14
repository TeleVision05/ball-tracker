"""Aimed synthetic deliveries with known ground truth, for the demo and the benchmark.

``synthetic.py`` samples a release velocity and lets the ball land wherever it
lands. Here a delivery is described the way a coach would describe it (pace,
line, length, swing, turn) and the release velocity is solved so the simulated
ball pitches close to that spot. The flight itself is always simulated by
:func:`balltracker.synthetic.simulate_trajectory`, and the ground truth is read
back from the simulated path rather than from the request.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .calibration import CameraCalibration, solve_pnp
from .landmarks import WARN_RMSE_PX
from .pitch import BALL_RADIUS_M, PITCH_LENGTH_M, stump_landmarks_3d
from .synthetic import GRAVITY_M_S2, CameraModel, DeliveryParams, simulate_trajectory
from .tracking import TrackPoint

SIM_DT_S = 1.0 / 1000.0


@dataclass(frozen=True)
class DeliverySpec:
    """A delivery described in cricket terms.

    ``line_m`` is the bounce offset from middle stump (+ = off side for a
    right-handed batter), ``length_m`` the bounce distance in front of the
    batter's stumps, ``swing_m_s2`` a constant sideways acceleration before the
    bounce and ``turn_m`` the simulator's sideways kick at the bounce (the
    sideways speed gained is three times this value, in m/s).
    """

    speed_kmh: float
    line_m: float
    length_m: float
    swing_m_s2: float = 0.0
    turn_m: float = 0.0
    release_pos_m: tuple[float, float, float] = (0.0, 1.2, 2.0)
    restitution: float = 0.75
    friction: float = 0.15


@dataclass(frozen=True)
class SimulatedDelivery:
    """The simulator's path for one delivery plus its ground-truth metrics."""

    spec: DeliverySpec
    params: DeliveryParams
    times_s: np.ndarray
    positions_m: np.ndarray
    bounce_index: int
    truth: dict


def protocol_camera(resolution: tuple[int, int] = (1280, 720)) -> CameraModel:
    """Behind-the-bowler camera framed as docs/CAPTURE_PROTOCOL.md asks.

    Same spot as ``synthetic.default_camera`` (4 m back, 1.66 m high, 55° field
    of view) but tilted down so both stump sets are fully in frame. The default
    camera leaves the bowler-end stump bases below the frame.
    """
    return CameraModel(
        position_m=np.array([0.0, -4.0, 1.66], dtype=np.float64),
        look_at_m=np.array([0.0, PITCH_LENGTH_M * 0.5, -0.1], dtype=np.float64),
        fov_deg=55.0,
        resolution=resolution,
    )


def _heading_deg(v_xy: np.ndarray) -> float:
    """Heading in the pitch plane, degrees, +Y = 0, matching ``metrics.py``."""
    return math.degrees(math.atan2(float(v_xy[0]), float(v_xy[1])))


def aim_release_velocity(spec: DeliverySpec) -> np.ndarray:
    """Release velocity that pitches the ball at ``(line_m, length_m)``.

    Uses the closed-form flight under gravity plus the constant swing
    acceleration, solving for the flight time whose speed matches the request.
    """
    x0, y0, z0 = spec.release_pos_m
    speed = spec.speed_kmh / 3.6
    target_x = spec.line_m
    target_y = PITCH_LENGTH_M - spec.length_m

    def velocity(t: float) -> np.ndarray:
        return np.array(
            [
                (target_x - x0 - 0.5 * spec.swing_m_s2 * t * t) / t,
                (target_y - y0) / t,
                (BALL_RADIUS_M - z0 + 0.5 * GRAVITY_M_S2 * t * t) / t,
            ],
            dtype=np.float64,
        )

    def speed_gap(t: float) -> float:
        return float(np.linalg.norm(velocity(t))) - speed

    # Speed falls with flight time until the lob branch takes over; keep the flat root.
    slowest = minimize_scalar(speed_gap, bounds=(0.05, 5.0), method="bounded")
    if speed_gap(slowest.x) > 0.0:
        raise ValueError("No release at this speed reaches the requested spot")
    t_flight = brentq(speed_gap, 0.05, float(slowest.x))
    return velocity(t_flight)


def _ground_truth(
    params: DeliveryParams, positions: np.ndarray, bounce_index: int, dt: float
) -> dict:
    """Read release speed, bounce spot, swing and turn back from the simulated path."""
    bounce = positions[bounce_index]
    v_in = (positions[bounce_index] - positions[bounce_index - 1])[:2] / dt
    v_out = (positions[bounce_index + 1] - positions[bounce_index])[:2] / dt
    v_release = params.release_vel_m_s
    return {
        "release_speed_kmh": float(np.linalg.norm(v_release) * 3.6),
        "bounce_line_m": float(bounce[0]),
        "bounce_length_m": float(PITCH_LENGTH_M - bounce[1]),
        "bounce_time_s": float(bounce_index * dt),
        "swing_deg": _heading_deg(v_in) - _heading_deg(v_release[:2]),
        "turn_deg": _heading_deg(v_out) - _heading_deg(v_in),
    }


def simulate_delivery(spec: DeliverySpec, dt: float = SIM_DT_S) -> SimulatedDelivery:
    """Aim and simulate one delivery; raise ``ValueError`` if it never bounces."""
    params = DeliveryParams(
        release_pos_m=np.array(spec.release_pos_m, dtype=np.float64),
        release_vel_m_s=aim_release_velocity(spec),
        beta_s=spec.swing_m_s2,
        restitution=spec.restitution,
        friction=spec.friction,
        turn_m=spec.turn_m,
        radar_speed_kmh=spec.speed_kmh,
    )
    times, positions, bounce_index, _ = simulate_trajectory(params, dt=dt)
    if bounce_index < 1 or bounce_index >= len(positions) - 1:
        raise ValueError("Simulated delivery did not bounce on the pitch")
    truth = _ground_truth(params, positions, bounce_index, dt)
    return SimulatedDelivery(spec, params, times, positions, bounce_index, truth)


def sample_positions(delivery: SimulatedDelivery, frame_times_s: np.ndarray) -> np.ndarray:
    """Ball centre at each camera frame time, linearly interpolated from the simulation."""
    times = delivery.times_s
    pos = delivery.positions_m
    return np.stack([np.interp(frame_times_s, times, pos[:, axis]) for axis in range(3)], axis=1)


def clicked_stump_points(
    camera: CameraModel, rng: np.random.Generator, click_noise_px: float
) -> dict[str, np.ndarray]:
    """Stump landmarks as a person would click them: visible ones only, plus click error."""
    landmarks = stump_landmarks_3d()
    uv = camera.project(np.stack(list(landmarks.values())))
    w, h = camera.resolution
    points: dict[str, np.ndarray] = {}
    for i, key in enumerate(landmarks):
        click = uv[i] + rng.normal(0.0, click_noise_px, size=2)
        if np.all(np.isfinite(uv[i])) and 0 <= uv[i][0] < w and 0 <= uv[i][1] < h:
            points[key] = click
    return points


@dataclass(frozen=True)
class ClickCalibration:
    """A calibration from simulated clicks and how many rounds of clicking it took."""

    calib: CameraCalibration
    points: dict[str, np.ndarray]
    attempts: int


def calibrate_from_clicks(
    camera: CameraModel,
    rng: np.random.Generator,
    click_noise_px: float,
    fov_deg: float,
    max_attempts: int = 5,
) -> ClickCalibration:
    """Calibrate as the app does: PnP on clicked stumps with the configured FOV.

    The app warns when the reprojection error is above ``WARN_RMSE_PX`` and asks
    for more careful clicks, so a fresh set of clicks is simulated in that case.
    The last attempt is returned even if it is still above the threshold.
    """
    for attempt in range(1, max_attempts + 1):
        points = clicked_stump_points(camera, rng, click_noise_px)
        calib = solve_pnp(points, image_size=camera.resolution, fov_deg=fov_deg)
        if calib.rms_reprojection_error <= WARN_RMSE_PX:
            break
    return ClickCalibration(calib, points, attempt)


def track_points(
    uv: np.ndarray, frame_indices: np.ndarray, fps: float, keep: np.ndarray | None = None
) -> list[TrackPoint]:
    """Turn per-frame pixel positions into ``TrackPoint`` objects, skipping dropped frames."""
    points: list[TrackPoint] = []
    for i, frame_idx in enumerate(frame_indices):
        if keep is not None and not keep[i]:
            continue
        u, v = float(uv[i, 0]), float(uv[i, 1])
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        points.append(
            TrackPoint(
                frame_idx=int(frame_idx),
                t_s=float(frame_idx) / fps,
                u=u,
                v=v,
                radius_px=4.0,
                confidence=1.0,
                predicted=False,
            )
        )
    return points
