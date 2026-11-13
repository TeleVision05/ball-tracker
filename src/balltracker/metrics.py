"""Delivery metrics derived from trajectory fits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .physics import TrajectoryFit
from .pitch import PITCH_LENGTH_M


@dataclass
class DeliveryMetrics:
    """Computed bowling delivery metrics."""

    release_speed_m_s: float
    release_speed_kmh: float
    bounce_speed_m_s: float | None
    beta_s_deg: float
    c_s: float
    lateral_drift_m: float
    turn_deg: float
    turn_m: float
    pitch_line_m: float
    pitch_length_m: float
    bounce_y_m: float
    confidence: dict[str, bool] = field(default_factory=dict)


def no_spin_rebound(
    v_in: np.ndarray,
    restitution: float = 0.75,
    friction: float = 0.15,
) -> np.ndarray:
    """Simple no-spin bounce model: reverse vz with restitution, damp horizontal."""
    v = np.asarray(v_in, dtype=np.float64).copy()
    v[2] = -restitution * v[2]
    v[0] *= 1.0 - friction
    v[1] *= 1.0 - friction
    return v


def _horizontal_heading(v: np.ndarray) -> float:
    """Heading angle in XY plane (radians), +Y = 0."""
    return float(math.atan2(v[0], v[1]))


def compute_metrics(
    fit: TrajectoryFit,
    restitution: float = 0.75,
    friction: float = 0.15,
) -> DeliveryMetrics:
    """Compute delivery metrics from a ``TrajectoryFit``."""
    v_release = np.asarray(fit.pre.v0, dtype=np.float64)
    release_speed = float(np.linalg.norm(v_release))
    release_speed_kmh = release_speed * 3.6

    bounce_pos = fit.bounce_pos_m
    if bounce_pos is None:
        bounce_pos = np.zeros(3, dtype=np.float64)

    bounce_y = float(bounce_pos[1])
    pitch_line = float(bounce_pos[0])
    pitch_length = float(PITCH_LENGTH_M - bounce_y)

    # Lateral drift: bounce x minus release heading line at bounce y.
    if abs(v_release[1]) > 1e-6:
        x_on_heading = fit.pre.x0[0] + (bounce_y - fit.pre.x0[1]) * v_release[0] / v_release[1]
    else:
        x_on_heading = fit.pre.x0[0]
    lateral_drift = float(bounce_pos[0] - x_on_heading)

    # Approach velocity at bounce from pre segment.
    if fit.bounce_time_s is not None:
        dt = fit.bounce_time_s - fit.pre.t0
        v_bounce_in = v_release + np.array([0.0, 0.0, -9.81]) * dt
    else:
        v_bounce_in = v_release.copy()

    beta_s_deg = float(
        math.degrees(_horizontal_heading(v_bounce_in) - _horizontal_heading(v_release))
    )

    # Grimshaw-style side-force coefficient approximation:
    # c_s ≈ 2 * lateral_drift / (path_length * mean_speed)
    path_length = float(np.linalg.norm(bounce_pos - fit.pre.x0))
    mean_speed = 0.5 * (release_speed + float(np.linalg.norm(v_bounce_in)))
    c_s = float(2.0 * lateral_drift / max(path_length * mean_speed, 1e-3))

    bounce_speed = float(np.linalg.norm(v_bounce_in))

    # Turn vs no-spin rebound model.
    v_no_spin = no_spin_rebound(v_bounce_in, restitution=restitution, friction=friction)
    if fit.post is not None:
        v_actual = np.asarray(fit.post.v0, dtype=np.float64)
    else:
        v_actual = v_no_spin

    turn_m = float(v_actual[0] - v_no_spin[0]) * 0.5  # rough lateral metres proxy
    turn_deg = float(math.degrees(_horizontal_heading(v_actual) - _horizontal_heading(v_no_spin)))

    confidence = {
        "trajectory_fit_ok": fit.reprojection_rmse < 15.0,
        "bounce_detected": fit.bounce_time_s is not None,
        "post_bounce_fit": fit.post is not None,
    }

    return DeliveryMetrics(
        release_speed_m_s=release_speed,
        release_speed_kmh=release_speed_kmh,
        bounce_speed_m_s=bounce_speed,
        beta_s_deg=beta_s_deg,
        c_s=c_s,
        lateral_drift_m=lateral_drift,
        turn_deg=turn_deg,
        turn_m=turn_m,
        pitch_line_m=pitch_line,
        pitch_length_m=pitch_length,
        bounce_y_m=bounce_y,
        confidence=confidence,
    )


def metrics_to_dict(m: DeliveryMetrics) -> dict:
    """Convert metrics to a JSON-serializable dict."""
    d = asdict(m)
    return d
