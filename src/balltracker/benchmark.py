"""Synthetic accuracy benchmark: many simulated deliveries through calibration, fit and metrics.

Each case samples a delivery (pace, line, length, swing, turn), a capture
setup (frame rate, detection noise, dropped frames, stray detections) and a
calibration made from noisy stump clicks. The simulated ball is projected into
the camera, the noise is applied to those pixels, and the result goes through
the same ``fit_trajectory`` and ``compute_metrics`` calls the video pipeline
uses. The image detector itself is not run here; ``balltracker demo`` runs it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import load_config
from .metrics import compute_metrics
from .physics import fit_trajectory
from .scenario import (
    DeliverySpec,
    SimulatedDelivery,
    calibrate_from_clicks,
    protocol_camera,
    sample_positions,
    simulate_delivery,
    track_points,
)
from .synthetic import CameraModel
from .tracking import TrackPoint, filter_ballistic_segment

DEFAULT_SEED = 20261113
DEFAULT_N = 500
FRAME_RATES = (60, 120, 240)
# pipeline: noisy detections, then the tracker's outlier filter (what a video gets).
# no_filter: the same noisy detections straight into the fit.
# noise_free: perfect detections and clicks straight into the fit (the fit's own error).
MODES = ("pipeline", "no_filter", "noise_free")

# Ranges per bowling style: speed km/h, swing m/s^2, turn (simulator units),
# release height m, length m in front of the batter's stumps.
BOWLING_STYLES: dict[str, dict] = {
    "fast": {
        "weight": 0.4,
        "speed": (130.0, 150.0),
        "swing": (-3.0, 3.0),
        "turn": (-0.12, 0.12),
        "release_z": (2.0, 2.3),
        "length": (1.5, 10.0),
    },
    "medium": {
        "weight": 0.25,
        "speed": (110.0, 130.0),
        "swing": (-3.0, 3.0),
        "turn": (-0.15, 0.15),
        "release_z": (1.9, 2.2),
        "length": (1.5, 9.0),
    },
    "spin": {
        "weight": 0.35,
        "speed": (70.0, 95.0),
        "swing": (-0.8, 0.8),
        "turn": (-0.9, 0.9),
        "release_z": (1.8, 2.1),
        "length": (1.0, 6.0),
    },
}

# (key, label, unit, scale from metres/km/h/deg to the reported unit)
METRICS: tuple[tuple[str, str, str, float], ...] = (
    ("release_speed_kmh", "Release speed", "km/h", 1.0),
    ("bounce_line_m", "Bounce line (sideways)", "cm", 100.0),
    ("bounce_length_m", "Bounce length", "cm", 100.0),
    ("swing_deg", "Swing before the bounce", "deg", 1.0),
    ("turn_deg", "Turn off the pitch", "deg", 1.0),
)


@dataclass(frozen=True)
class CaptureConditions:
    """How the simulated camera and detector see one delivery."""

    fps: int
    pixel_noise_px: float
    dropout_rate: float
    burst_frames: int
    hidden_release_frames: int
    outlier_rate: float
    click_noise_px: float
    frame_phase: float


def sample_spec(rng: np.random.Generator) -> tuple[str, DeliverySpec]:
    """Pick a bowling style, then a delivery inside its ranges."""
    names = list(BOWLING_STYLES)
    weights = np.array([BOWLING_STYLES[n]["weight"] for n in names])
    style = names[int(rng.choice(len(names), p=weights / weights.sum()))]
    r = BOWLING_STYLES[style]
    spec = DeliverySpec(
        speed_kmh=float(rng.uniform(*r["speed"])),
        line_m=float(rng.uniform(-0.35, 0.45)),
        length_m=float(rng.uniform(*r["length"])),
        swing_m_s2=float(rng.uniform(*r["swing"])),
        turn_m=float(rng.uniform(*r["turn"])),
        release_pos_m=(
            float(rng.uniform(-0.45, 0.45)),
            float(rng.uniform(0.8, 1.4)),
            float(rng.uniform(*r["release_z"])),
        ),
        restitution=float(rng.uniform(0.65, 0.85)),
        friction=float(rng.uniform(0.05, 0.25)),
    )
    return style, spec


def sample_conditions(rng: np.random.Generator) -> CaptureConditions:
    """Phone-like capture: 60/120/240 fps, 0.5 to 2 px jitter, missed and stray detections."""
    return CaptureConditions(
        fps=int(rng.choice(FRAME_RATES)),
        pixel_noise_px=float(rng.uniform(0.5, 2.0)),
        dropout_rate=float(rng.uniform(0.0, 0.15)),
        burst_frames=int(rng.integers(0, 5)) if rng.random() < 0.5 else 0,
        hidden_release_frames=int(rng.integers(0, 4)),
        outlier_rate=0.02,
        click_noise_px=1.5,
        frame_phase=float(rng.random()),
    )


def observe(
    delivery: SimulatedDelivery,
    camera: CameraModel,
    cond: CaptureConditions,
    rng: np.random.Generator,
    use_filter: bool = True,
) -> list[TrackPoint]:
    """Simulated detector output: projected ball centres with noise, gaps and strays.

    With ``use_filter`` the points then go through the tracker's own outlier
    filter, as every video tracking method does before the fit.
    """
    n_frames = int(delivery.times_s[-1] * cond.fps)
    frames = np.arange(n_frames)
    times = (frames + cond.frame_phase) / cond.fps
    uv = camera.project(sample_positions(delivery, times))
    uv = uv + rng.normal(0.0, cond.pixel_noise_px, size=uv.shape)

    strays = rng.random(n_frames) < cond.outlier_rate
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_frames)
    radii = rng.uniform(5.0, 15.0, size=n_frames)
    uv[strays, 0] += radii[strays] * np.cos(angles[strays])
    uv[strays, 1] += radii[strays] * np.sin(angles[strays])

    keep = rng.random(n_frames) >= cond.dropout_rate
    keep[: cond.hidden_release_frames] = False
    if cond.burst_frames:
        # Small, far-away ball: gaps cluster in the second half of the flight.
        start = int(rng.integers(n_frames // 2, max(n_frames // 2 + 1, n_frames - 5)))
        keep[start : start + cond.burst_frames] = False

    w, h = camera.resolution
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    track = track_points(uv, frames, float(cond.fps), keep & inside)
    return filter_ballistic_segment(track) if use_filter else track


def estimate(track: list[TrackPoint], calib, fps: float, physics: dict) -> dict:
    """Fit and measure exactly as ``pipeline.analyze_delivery`` does."""
    fit = fit_trajectory(
        track,
        calib,
        fps=fps,
        g=float(physics.get("gravity_m_s2", 9.81)),
        use_drag=float(physics.get("drag_coefficient", 0.0)) > 0,
    )
    m = compute_metrics(
        fit,
        restitution=float(physics.get("restitution", 0.75)),
        friction=float(physics.get("friction_coefficient", 0.15)),
    )
    return {
        "release_speed_kmh": m.release_speed_kmh,
        "bounce_line_m": m.pitch_line_m,
        "bounce_length_m": m.pitch_length_m,
        "swing_deg": m.beta_s_deg,
        "turn_deg": m.turn_deg,
    }


def _simulate_case(rng: np.random.Generator) -> tuple[str, SimulatedDelivery]:
    """Sample until the delivery is physically possible and bounces on the pitch."""
    for _ in range(50):
        style, spec = sample_spec(rng)
        try:
            return style, simulate_delivery(spec)
        except ValueError:
            continue
    raise RuntimeError("Could not sample a delivery that bounces on the pitch")


def noise_free(cond: CaptureConditions) -> CaptureConditions:
    """Same frame rate and timing, but perfect detections and perfect stump clicks."""
    return replace(
        cond,
        pixel_noise_px=0.0,
        dropout_rate=0.0,
        burst_frames=0,
        hidden_release_frames=0,
        outlier_rate=0.0,
        click_noise_px=0.0,
    )


def run_case(
    index: int, seed: int, camera: CameraModel, config: dict, mode: str = "pipeline"
) -> dict:
    """One benchmark case. Case ``i`` depends only on ``(seed, i)``.

    ``mode`` is one of :data:`MODES`. All three see the same delivery and frame
    rate; ``no_filter`` also sees the same noisy detections as ``pipeline``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rng = np.random.default_rng([seed, index])
    style, delivery = _simulate_case(rng)
    cond = sample_conditions(rng)
    if mode == "noise_free":
        cond = noise_free(cond)
    fov = float(config["camera"]["fov_deg"])
    clicked = calibrate_from_clicks(camera, rng, cond.click_noise_px, fov)
    calib = clicked.calib
    track = observe(delivery, camera, cond, rng, use_filter=mode == "pipeline")
    row = {
        "case": index,
        "mode": mode,
        "style": style,
        "fps": cond.fps,
        "pixel_noise_px": cond.pixel_noise_px,
        "track_points": len(track),
        "calib_rmse_px": float(calib.rms_reprojection_error),
        "calib_attempts": clicked.attempts,
        "truth": dict(delivery.truth),
    }
    try:
        row["estimate"] = estimate(track, calib, float(cond.fps), config.get("physics", {}))
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        row["failed"] = type(exc).__name__
        return row
    row["error"] = {key: row["estimate"][key] - row["truth"][key] for key, *_ in METRICS}
    return row


def run_benchmark(
    n: int = DEFAULT_N, seed: int = DEFAULT_SEED, mode: str = "pipeline"
) -> list[dict]:
    """Run ``n`` seeded cases on a 720p behind-the-bowler camera framed per the protocol."""
    config = load_config()
    camera = protocol_camera()
    return [run_case(i, seed, camera, config, mode=mode) for i in range(n)]


def error_stats(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Median and 90th percentile of absolute error, and median signed error, per metric."""
    done = [r for r in rows if "error" in r]
    stats: dict[str, dict[str, float]] = {}
    for key, _label, _unit, scale in METRICS:
        signed = np.array([r["error"][key] for r in done], dtype=np.float64) * scale
        if signed.size == 0:
            stats[key] = {"median": float("nan"), "p90": float("nan"), "bias": float("nan")}
            continue
        absolute = np.abs(signed)
        stats[key] = {
            "median": float(np.median(absolute)),
            "p90": float(np.percentile(absolute, 90)),
            "bias": float(np.median(signed)),
        }
    return stats
