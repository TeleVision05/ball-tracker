"""Turn an Android capture sidecar and clip into delivery stats."""

from __future__ import annotations

import math
from pathlib import Path

from .capture_sidecar import (
    CaptureSidecar,
    apply_presentation_times,
    calibrate_from_capture,
    load_capture_sidecar,
)
from .config import load_config
from .metrics import compute_metrics, metrics_to_dict
from .physics import fit_trajectory
from .plain_language import summarize_delivery
from .tracking import TrackPoint, track_video

_NO_TIMESTAMPS = "No presentation timestamps; times use frame index divided by fps."
_PARTIAL_TIMESTAMPS = "Some track frames have no presentation timestamp."
_FPS_MISMATCH = "Measured fps differs from the requested capture fps."


def resolve_capture_video(
    sidecar: CaptureSidecar,
    sidecar_path: Path,
    video_path: Path | None,
) -> Path:
    """Locate the clip named by the sidecar, or use an explicit video path."""
    if video_path is not None:
        resolved = Path(video_path)
    else:
        resolved = Path(sidecar_path).parent / sidecar.videoFile
    if not resolved.exists():
        raise FileNotFoundError(
            f"Sidecar was read but the video file is required: {resolved}"
        )
    return resolved


def analyze_capture(
    sidecar_path: str | Path,
    video_path: str | Path | None = None,
    detection_method: str = "auto",
    config_path: str | Path | None = None,
) -> dict:
    """Track a capture clip and return delivery metrics, fit, and timing notes."""
    sidecar_path = Path(sidecar_path)
    config = load_config(config_path)
    sidecar = load_capture_sidecar(sidecar_path)
    calib = calibrate_from_capture(sidecar)
    video = resolve_capture_video(
        sidecar,
        sidecar_path,
        None if video_path is None else Path(video_path),
    )
    fps = _tracker_fps(sidecar, config)
    track = track_video(video, fps=fps, config=config, method=detection_method)
    timestamps = sidecar.presentationTimestampsUs
    if timestamps:
        track = apply_presentation_times(track, timestamps)
    if len(track) < 5:
        raise RuntimeError(f"Insufficient track points ({len(track)}) in {video}")

    physics = config.get("physics", {})
    g = float(physics.get("gravity_m_s2", 9.81))
    use_drag = float(physics.get("drag_coefficient", 0.0)) > 0
    fit = fit_trajectory(track, calib, fps=fps, g=g, use_drag=use_drag)
    metrics = compute_metrics(
        fit,
        restitution=float(physics.get("restitution", 0.75)),
        friction=float(physics.get("friction_coefficient", 0.15)),
    )
    metrics_dict = metrics_to_dict(metrics)
    center = calib.camera_center_world
    return {
        "capture_id": str(sidecar.id),
        "video_path": str(video),
        "metrics": metrics_dict,
        "summary": summarize_delivery(metrics_dict),
        "track": [
            {
                "frame_idx": point.frame_idx,
                "t_s": point.t_s,
                "u": point.u,
                "v": point.v,
                "radius_px": point.radius_px,
                "confidence": point.confidence,
                "predicted": point.predicted,
            }
            for point in track
        ],
        "fit": {
            "reprojection_rmse_px": fit.reprojection_rmse,
            "bounce_time_s": fit.bounce_time_s,
            "bounce_position_m": (
                fit.bounce_pos_m.tolist() if fit.bounce_pos_m is not None else None
            ),
            "release_position_m": fit.pre.x0.tolist(),
            "release_velocity_m_s": fit.pre.v0.tolist(),
        },
        "calib_rmse_px": float(calib.rms_reprojection_error),
        "camera_center_m": [float(center[0]), float(center[1]), float(center[2])],
        "timing_source": _timing_source(timestamps, track),
        "warnings": _warnings(sidecar, track),
    }


def _positive_finite(value: float | None) -> bool:
    if value is None:
        return False
    number = float(value)
    return math.isfinite(number) and number > 0.0


def _tracker_fps(sidecar: CaptureSidecar, config: dict) -> float:
    if _positive_finite(sidecar.measuredFps):
        return float(sidecar.measuredFps)
    if _positive_finite(sidecar.captureFps):
        return float(sidecar.captureFps)
    return float(config["camera"]["fps"])


def _frame_in_timestamps(point: TrackPoint, timestamps: list[int]) -> bool:
    return 0 <= point.frame_idx < len(timestamps)


def _timing_source(timestamps: list[int], track: list[TrackPoint]) -> str:
    if timestamps and all(_frame_in_timestamps(point, timestamps) for point in track):
        return "presentation_timestamps"
    return "frame_index"


def _warnings(sidecar: CaptureSidecar, track: list[TrackPoint]) -> list[str]:
    warnings: list[str] = []
    timestamps = sidecar.presentationTimestampsUs
    if not timestamps:
        warnings.append(_NO_TIMESTAMPS)
    elif any(not _frame_in_timestamps(point, timestamps) for point in track):
        warnings.append(_PARTIAL_TIMESTAMPS)
    measured = sidecar.measuredFps
    capture = float(sidecar.captureFps)
    if (
        _positive_finite(measured)
        and _positive_finite(capture)
        and abs(float(measured) - capture) / capture > 0.05
    ):
        warnings.append(_FPS_MISMATCH)
    return warnings
