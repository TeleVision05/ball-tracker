"""End-to-end delivery analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calib_io import load_calibration
from .calibration import (
    CameraCalibration,
    project_stumps_from_synthetic_camera,
    solve_pnp,
)
from .config import load_config
from .errors import CalibrationMissingError, NoBallFoundError
from .metrics import compute_metrics, metrics_to_dict
from .physics import fit_trajectory
from .pitch import stump_landmarks_3d
from .result_checks import result_checks
from .synthetic import CameraModel
from .tracking import TrackPoint, track_video


_TRACK_HINT = (
    "The hybrid tracker (method='auto') seeds its search from a "
    "MediaPipe-estimated release point and hand/arm motion — if the "
    "bowler isn't clearly visible early in the window, that seed can "
    "be missing or off. Try widening the trim (a few extra frames "
    "before release and after the bounce), or fall back to detection "
    "method 'white'/'hsv' if the ball's colour contrasts cleanly "
    "with the background."
)


def _companion_json_path(video_path: Path) -> Path:
    return video_path.with_suffix(".json")


def _track_from_gt_json(gt: dict) -> list[TrackPoint]:
    """Build a track from companion ``track_2d_px`` (labeled ground truth)."""
    fps = float(gt.get("fps", 120))
    track_2d = gt["track_2d_px"]
    points: list[TrackPoint] = []
    for i, uv in enumerate(track_2d):
        if uv is None or len(uv) < 2:
            continue
        u, v = float(uv[0]), float(uv[1])
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        points.append(
            TrackPoint(
                frame_idx=i,
                t_s=i / fps,
                u=u,
                v=v,
                radius_px=6.0,
                confidence=1.0,
                predicted=False,
            )
        )
    return points


def load_gt_json(json_path: str | Path) -> dict:
    """Load ground-truth / synthetic annotation JSON."""
    with Path(json_path).open(encoding="utf-8") as f:
        return json.load(f)


def calibration_from_gt_json(gt: dict, config: dict | None = None) -> CameraCalibration | None:
    """Build calibration from GT JSON if camera fields are present."""
    if config is None:
        config = load_config()

    if "camera" in gt:
        cam_data = gt["camera"]
        camera = CameraModel(
            position_m=np.array(cam_data["position_m"], dtype=np.float64),
            look_at_m=np.array(cam_data["look_at_m"], dtype=np.float64),
            fov_deg=float(cam_data.get("fov_deg", config["camera"]["fov_deg"])),
            resolution=tuple(cam_data.get("resolution", config["camera"]["resolution"])),
        )
        return project_stumps_from_synthetic_camera(camera)

    if all(k in gt for k in ("camera_position_m", "camera_look_at_m")):
        camera = CameraModel(
            position_m=np.array(gt["camera_position_m"], dtype=np.float64),
            look_at_m=np.array(gt["camera_look_at_m"], dtype=np.float64),
            fov_deg=float(gt.get("camera_fov_deg", config["camera"]["fov_deg"])),
            resolution=tuple(gt.get("camera_resolution", config["camera"]["resolution"])),
        )
        return project_stumps_from_synthetic_camera(camera)

    return None


def load_calibration_file(path: str | Path) -> CameraCalibration:
    """Load a saved calibration JSON file."""
    calib, _ = load_calibration(path)
    return calib


def _find_adjacent_calib_file(video_path: Path) -> Path | None:
    """Locate ``video.calib.json`` or ``session.calib.json`` in the video folder."""
    per_video = video_path.with_suffix(".calib.json")
    if per_video.exists():
        return per_video
    session_calib = video_path.parent / "session.calib.json"
    if session_calib.exists():
        return session_calib
    return None


def resolve_calibration(
    video_path: Path,
    calib: CameraCalibration | None,
    stump_image_points: dict | None,
    config: dict,
    calib_path: str | Path | None = None,
) -> CameraCalibration:
    """Resolve camera calibration for real-video analysis.

    Order: explicit calib → stump_image_points → calib_path → adjacent calib
    files → companion JSON camera fields. Raises if none found.
    """
    if calib is not None:
        return calib

    if stump_image_points:
        w, h = config["camera"]["resolution"]
        return solve_pnp(stump_image_points, image_size=(int(w), int(h)))

    if calib_path is not None:
        path = Path(calib_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        return load_calibration_file(path)

    adjacent = _find_adjacent_calib_file(video_path)
    if adjacent is not None:
        return load_calibration_file(adjacent)

    companion = _companion_json_path(video_path)
    if companion.exists():
        gt = load_gt_json(companion)
        calib_from_json = calibration_from_gt_json(gt, config)
        if calib_from_json is not None:
            return calib_from_json

    raise CalibrationMissingError(
        f"No camera calibration found for {video_path}. "
        "Run: python scripts/calibrate_stumps.py --video <path> --out <path.calib.json>"
    )


def analyze_delivery(
    video_path: str | Path,
    calib: CameraCalibration | None = None,
    stump_image_points: dict | None = None,
    calib_path: str | Path | None = None,
    config_path: str | Path | None = None,
    detection_method: str = "auto",
    use_companion_track: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict:
    """Run full analysis pipeline on one delivery video.

    Restrict tracking to ``[start_frame, end_frame]`` to avoid false motion
    detections outside the actual delivery (walk-ups, idle nets, etc.).
    """
    video_path = Path(video_path)
    config = load_config(config_path)
    calib = resolve_calibration(
        video_path,
        calib,
        stump_image_points,
        config,
        calib_path=calib_path,
    )

    companion = _companion_json_path(video_path)
    fps = None
    gt: dict | None = None
    if companion.exists():
        gt = load_gt_json(companion)
        fps = gt.get("fps")

    import cv2

    # Prefer container FPS for real videos.
    cap = cv2.VideoCapture(str(video_path))
    container_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    fps_val = float(
        fps if fps else (container_fps if container_fps > 1 else config["camera"]["fps"])
    )

    if end_frame is None:
        end_frame = max(0, total_frames - 1)
    start_frame = int(max(0, start_frame))
    end_frame = int(min(end_frame, max(0, total_frames - 1)))

    track: list[TrackPoint]
    if use_companion_track and gt is not None and "track_2d_px" in gt:
        track = [tp for tp in _track_from_gt_json(gt) if start_frame <= tp.frame_idx <= end_frame]
    else:
        track = track_video(
            video_path,
            fps=fps_val,
            config=config,
            method=detection_method,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    if len(track) < 5:
        raise NoBallFoundError(
            f"Insufficient track points ({len(track)}) in {video_path} "
            f"for frames [{start_frame}, {end_frame}]. {_TRACK_HINT}"
        )
    return _delivery_result(video_path, track, calib, config, fps_val, start_frame, end_frame)


def analyze_from_track(
    video_path: str | Path,
    track: list[TrackPoint],
    calib: CameraCalibration | None = None,
    stump_image_points: dict | None = None,
    calib_path: str | Path | None = None,
    config_path: str | Path | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> dict:
    """Fit metrics from an existing 2D track (e.g. manual clicks).

    Same output schema as :func:`analyze_delivery`, but skips automatic
    detection/tracking.
    """
    video_path = Path(video_path)
    config = load_config(config_path)
    calib = resolve_calibration(
        video_path,
        calib,
        stump_image_points,
        config,
        calib_path=calib_path,
    )

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    container_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    fps_val = float(container_fps if container_fps > 1 else config["camera"]["fps"])

    if start_frame is None and track:
        start_frame = min(tp.frame_idx for tp in track)
    if end_frame is None and track:
        end_frame = max(tp.frame_idx for tp in track)
    start_frame = int(max(0, start_frame or 0))
    end_frame = int(
        min(
            end_frame if end_frame is not None else max(0, total_frames - 1),
            max(0, total_frames - 1),
        )
    )

    if len(track) < 5:
        raise NoBallFoundError(
            f"Insufficient track points ({len(track)}). "
            "Mark the ball on at least 5 frames, then calculate."
        )

    result = _delivery_result(video_path, track, calib, config, fps_val, start_frame, end_frame)
    result["manual_track"] = True
    return result


def _delivery_result(
    video_path: Path,
    track: list[TrackPoint],
    calib: CameraCalibration,
    config: dict,
    fps: float,
    start_frame: int,
    end_frame: int,
) -> dict:
    """Fit the track and build the result dict both entry points return.

    ``checks`` holds plain-language warnings when the result looks wrong
    (see :mod:`balltracker.result_checks`); it is empty when nothing does.
    """
    physics = config.get("physics", {})
    g = float(physics.get("gravity_m_s2", 9.81))
    use_drag = float(physics.get("drag_coefficient", 0.0)) > 0
    fit = fit_trajectory(track, calib, fps=fps, g=g, use_drag=use_drag)
    metrics = metrics_to_dict(
        compute_metrics(
            fit,
            restitution=float(physics.get("restitution", 0.75)),
            friction=float(physics.get("friction_coefficient", 0.15)),
        )
    )
    return {
        "video_path": str(video_path),
        "metrics": metrics,
        "track": [_track_point_dict(tp) for tp in track],
        "fit": {
            "reprojection_rmse_px": fit.reprojection_rmse,
            "bounce_time_s": fit.bounce_time_s,
            "bounce_position_m": fit.bounce_pos_m.tolist()
            if fit.bounce_pos_m is not None
            else None,
            "release_position_m": fit.pre.x0.tolist(),
            "release_velocity_m_s": fit.pre.v0.tolist(),
        },
        "frame_window": {"start_frame": start_frame, "end_frame": end_frame},
        "calib_rmse_px": calib.rms_reprojection_error,
        "stump_projections": _stump_projections(calib),
        "fps": fps,
        "checks": result_checks(metrics, fps=fps),
    }


def _track_point_dict(tp: TrackPoint) -> dict:
    return {
        "frame_idx": tp.frame_idx,
        "t_s": tp.t_s,
        "u": tp.u,
        "v": tp.v,
        "radius_px": tp.radius_px,
        "confidence": tp.confidence,
        "predicted": tp.predicted,
    }


def _stump_projections(calib: CameraCalibration) -> dict[str, list[float]]:
    projections = {}
    for key, pt in stump_landmarks_3d().items():
        uv = calib.K @ (calib.R @ pt + calib.t)
        if abs(uv[2]) > 1e-9:
            projections[key] = (uv[:2] / uv[2]).tolist()
    return projections


def evaluate_against_gt(metrics: dict, gt_json_path: str | Path) -> dict:
    """Compute absolute errors against ground-truth JSON."""
    gt = load_gt_json(gt_json_path)
    m = metrics if "release_speed_kmh" in metrics else metrics.get("metrics", metrics)

    errors: dict[str, float | None] = {}

    if "radar_speed_kmh" in gt and "release_speed_kmh" in m:
        errors["release_speed_kmh_abs_err"] = abs(m["release_speed_kmh"] - gt["radar_speed_kmh"])

    if "bounce_line_m" in gt and "pitch_line_m" in m:
        errors["bounce_line_m_abs_err"] = abs(m["pitch_line_m"] - gt["bounce_line_m"])

    if "bounce_length_m" in gt and "pitch_length_m" in m:
        errors["pitch_length_m_abs_err"] = abs(m["pitch_length_m"] - gt["bounce_length_m"])

    if "bounce_position_m" in gt and "bounce_y_m" in m:
        gt_bounce = np.array(gt["bounce_position_m"], dtype=np.float64)
        est_bounce = np.array(
            [m.get("pitch_line_m", 0.0), m.get("bounce_y_m", 0.0), gt_bounce[2]],
            dtype=np.float64,
        )
        errors["bounce_position_m_abs_err"] = float(np.linalg.norm(est_bounce - gt_bounce))

    if "beta_s" in gt and "beta_s_deg" in m:
        errors["beta_s_abs_err"] = abs(m["beta_s_deg"] - gt["beta_s"])

    return errors
