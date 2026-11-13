"""``balltracker demo``: one synthetic delivery through the whole pipeline, no footage needed.

1. Simulate a delivery and render it as a 1080p, 120 fps behind-the-bowler clip.
2. Calibrate the camera with PnP on stump clicks (simulated with click error).
3. Run ``pipeline.analyze_delivery`` on the clip, the call behind
   ``scripts/analyze_delivery.py``: colour tracking, bounce-segmented fit, metrics.
4. Write an overlay video, a pitch map and ``summary.json``; print estimate
   against the simulator's truth.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .calib_io import save_calibration
from .calibration import CameraCalibration, project_points
from .config import load_config
from .demo_media import (
    OverlayPlan,
    fitted_pixels_by_frame,
    make_web_playable,
    render_overlay_video,
    save_pitch_map,
    view_around,
)
from .demo_scene import render_delivery_clip
from .physics import TrajectoryFit, fit_trajectory
from .pipeline import analyze_delivery
from .plain_language import summarize_delivery
from .scenario import (
    ClickCalibration,
    DeliverySpec,
    SimulatedDelivery,
    calibrate_from_clicks,
    protocol_camera,
    sample_positions,
    simulate_delivery,
)
from .synthetic import CameraModel
from .tracking import TrackPoint

DEMO_SPEC = DeliverySpec(
    speed_kmh=134.0,
    line_m=0.12,
    length_m=5.5,
    swing_m_s2=1.2,
    turn_m=0.08,
    release_pos_m=(0.25, 1.2, 2.1),
)
DEMO_SEED = 7
DEMO_FPS = 120
DEMO_RESOLUTION = (1920, 1080)
CLICK_NOISE_PX = 1.5
PREROLL_FRAMES = 12
POSTROLL_FRAMES = 12
# The default floor (0.4) drops a ball under about 4 px radius. At 1080p the
# ball near the batter is about 3 px, so the demo config lowers it.
DEMO_CONFIDENCE_FLOOR = 0.2
METHOD = "hsv"


def demo_camera() -> CameraModel:
    """Behind-the-bowler camera at 1080p, both stump sets in frame."""
    return protocol_camera(DEMO_RESOLUTION)


def write_demo_config(out_dir: Path) -> Path:
    """Default config with the detector's confidence floor lowered for a small far ball."""
    config = load_config()
    config["detection"]["confidence_threshold"] = DEMO_CONFIDENCE_FLOOR
    config["camera"]["resolution"] = list(DEMO_RESOLUTION)
    config["camera"]["fps"] = DEMO_FPS
    path = out_dir / "demo_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def simulate_clip(out_dir: Path, camera: CameraModel) -> tuple[SimulatedDelivery, Path, int]:
    """Simulate the demo delivery and write ``delivery.mp4``."""
    delivery = simulate_delivery(DEMO_SPEC)
    n_flight = int(delivery.times_s[-1] * DEMO_FPS) + 1
    positions = sample_positions(delivery, np.arange(n_flight) / DEMO_FPS)
    clip = out_dir / "delivery.mp4"
    written = render_delivery_clip(
        camera, positions, DEMO_FPS, clip, PREROLL_FRAMES, POSTROLL_FRAMES
    )
    return delivery, clip, written


def calibrate(
    out_dir: Path, camera: CameraModel, rng: np.random.Generator
) -> tuple[ClickCalibration, Path]:
    """PnP from simulated stump clicks, saved like ``scripts/calibrate_stumps.py`` does."""
    fov = float(load_config()["camera"]["fov_deg"])
    clicked = calibrate_from_clicks(camera, rng, CLICK_NOISE_PX, fov)
    path = out_dir / "session.calib.json"
    save_calibration(clicked.calib, path, stump_image_points=clicked.points)
    return clicked, path


def refit(result: dict, calib: CameraCalibration, config: dict) -> TrajectoryFit:
    """Re-run the pipeline's fit on its own track to get the curve for drawing.

    Same track, calibration and settings as ``analyze_delivery``, so the fit is
    the one behind the reported metrics.
    """
    track = [TrackPoint(**point) for point in result["track"]]
    physics = config.get("physics", {})
    return fit_trajectory(
        track,
        calib,
        fps=DEMO_FPS,
        g=float(physics.get("gravity_m_s2", 9.81)),
        use_drag=float(physics.get("drag_coefficient", 0.0)) > 0,
    )


def estimate_from(metrics: dict) -> dict:
    """Metrics renamed to the ground-truth keys."""
    return {
        "release_speed_kmh": float(metrics["release_speed_kmh"]),
        "bounce_line_m": float(metrics["pitch_line_m"]),
        "bounce_length_m": float(metrics["pitch_length_m"]),
        "swing_deg": float(metrics["beta_s_deg"]),
        "turn_deg": float(metrics["turn_deg"]),
    }


def _truth_with_clip_time(delivery: SimulatedDelivery) -> dict:
    truth = dict(delivery.truth)
    truth["bounce_frame"] = int(round(truth["bounce_time_s"] * DEMO_FPS)) + PREROLL_FRAMES
    return truth


def comparison_rows(estimate: dict, truth: dict) -> list[tuple[str, str, str]]:
    """(label, estimate, truth) rows in plain units."""

    def side(x: float) -> str:
        return f"{abs(x) * 100:.0f} cm {'off' if x >= 0 else 'leg'} side"

    return [
        (
            "Release speed",
            f"{estimate['release_speed_kmh']:.1f} km/h",
            f"{truth['release_speed_kmh']:.1f} km/h",
        ),
        (
            "Bounce length",
            f"{estimate['bounce_length_m']:.2f} m from stumps",
            f"{truth['bounce_length_m']:.2f} m",
        ),
        ("Bounce line", side(estimate["bounce_line_m"]), side(truth["bounce_line_m"])),
        ("Swing", f"{estimate['swing_deg']:+.1f} deg", f"{truth['swing_deg']:+.1f} deg"),
        ("Turn", f"{estimate['turn_deg']:+.1f} deg", f"{truth['turn_deg']:+.1f} deg"),
    ]


def overlay_plan(
    result: dict, fit: TrajectoryFit, calib: CameraCalibration, estimate: dict, truth: dict
) -> OverlayPlan:
    """Detections, fitted path, bounce mark and HUD text for the overlay video."""
    frames = np.array([p["frame_idx"] for p in result["track"]], dtype=int)
    uv = np.array([[p["u"], p["v"]] for p in result["track"]], dtype=np.float64)
    bounce_frame = None
    bounce_uv = None
    if fit.bounce_time_s is not None and fit.bounce_pos_m is not None:
        bounce_frame = int(round(fit.bounce_time_s * DEMO_FPS))
        bu, bv = project_points(calib, fit.bounce_pos_m.reshape(1, 3))[0]
        bounce_uv = (float(bu), float(bv))
    results = [
        f"{label}: {est}  (true {tru})"
        for label, est, tru in comparison_rows(estimate, truth)
        if label != "Swing"
    ]
    fit_uv = fitted_pixels_by_frame(fit, calib, int(frames.min()), int(frames.max()), DEMO_FPS)
    shown = np.vstack([uv, np.array(list(fit_uv.values()), dtype=np.float64)])
    shown = shown[np.all(np.isfinite(shown), axis=1)]
    return OverlayPlan(
        track_frames=frames,
        track_uv=uv,
        fit_uv_by_frame=fit_uv,
        bounce_frame=bounce_frame,
        bounce_uv=bounce_uv,
        flight_lines=[
            "BallTracker demo: synthetic delivery, 4x slow motion",
            "yellow: ball detections   blue: fitted flight   red x: bounce",
        ],
        result_lines=["Estimate (simulator truth)", *results],
        view=view_around(shown, DEMO_RESOLUTION, 16.0 / 9.0),
    )


def run_demo(out_dir: Path, seed: int = DEMO_SEED) -> dict:
    """Run the demo into ``out_dir`` and return the summary written to ``summary.json``."""
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    camera = demo_camera()
    rng = np.random.default_rng(seed)

    delivery, clip, frames_written = simulate_clip(out_dir, camera)
    clicked, calib_path = calibrate(out_dir, camera, rng)
    calib = clicked.calib
    config_path = write_demo_config(out_dir)
    result = analyze_delivery(
        clip, calib_path=calib_path, config_path=config_path, detection_method=METHOD
    )
    estimate = estimate_from(result["metrics"])
    truth = _truth_with_clip_time(delivery)

    fit = refit(result, calib, load_config(config_path))
    overlay = render_overlay_video(
        clip, out_dir / "overlay.mp4", overlay_plan(result, fit, calib, estimate, truth)
    )
    is_h264, codec = make_web_playable(overlay)
    pitch_map = save_pitch_map(estimate, truth, out_dir / "pitch_map.png")
    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = build_summary(
        seed, delivery, truth, estimate, result, frames_written, clicked, is_h264, codec
    )
    summary["files"] = {
        "clip": clip.name,
        "calibration": calib_path.name,
        "config": config_path.name,
        "analysis": "analysis.json",
        "overlay_video": overlay.name,
        "pitch_map": pitch_map.name,
    }
    summary["runtime_s"] = round(time.perf_counter() - started, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_summary(
    seed: int,
    delivery: SimulatedDelivery,
    truth: dict,
    estimate: dict,
    result: dict,
    frames_written: int,
    clicked: ClickCalibration,
    is_h264: bool,
    codec: str | None,
) -> dict:
    """The JSON summary: what was simulated, what was measured, and how far apart."""
    flight_frames = frames_written - PREROLL_FRAMES - POSTROLL_FRAMES
    calib = clicked.calib
    return {
        "what": "One synthetic delivery analysed like a real clip. Synthetic only.",
        "seed": seed,
        "delivery_requested": dataclasses.asdict(delivery.spec),
        "clip": {
            "fps": DEMO_FPS,
            "resolution": list(DEMO_RESOLUTION),
            "frames": frames_written,
            "frames_with_ball": flight_frames,
        },
        "calibration": {
            "method": "PnP on simulated stump clicks",
            "click_noise_px": CLICK_NOISE_PX,
            "points_clicked": len(clicked.points),
            "click_rounds": clicked.attempts,
            "rmse_px": float(calib.rms_reprojection_error),
        },
        "tracking": {
            "method": METHOD,
            "confidence_floor": DEMO_CONFIDENCE_FLOOR,
            "points": len(result["track"]),
        },
        "fit_reprojection_rmse_px": float(result["fit"]["reprojection_rmse_px"]),
        "truth": truth,
        "estimate": estimate,
        "error": {key: estimate[key] - truth[key] for key in estimate},
        "plain_language": summarize_delivery(result["metrics"]),
        "overlay_codec": codec,
        "overlay_is_h264": is_h264,
    }
