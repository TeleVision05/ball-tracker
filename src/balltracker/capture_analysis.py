"""Turn an Android capture sidecar and clip into delivery stats."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import cv2

from .calibration import CameraCalibration
from .capture_sidecar import (
    CaptureSidecar,
    apply_presentation_times,
    calibrate_from_capture,
    load_capture_sidecar,
)
from .errors import NoBallFoundError, VideoOpenError
from .config import load_config
from .metrics import compute_metrics, metrics_to_dict
from .physics import fit_trajectory
from .plain_language import summarize_delivery
from .result_checks import result_checks
from .tracking import TrackPoint, track_video

_NO_TIMESTAMPS = "No presentation timestamps; times use frame index divided by fps."
_PARTIAL_TIMESTAMPS = "Some track frames have no presentation timestamp."
_CLOCK_MISMATCH = (
    "Container timestamps do not match the wall clock; times use frame index divided by fps."
)
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
        raise FileNotFoundError(f"Sidecar was read but the video file is required: {resolved}")
    return resolved


def upright_video(
    video_path: Path,
    orientation_hint_degrees: int,
) -> tuple[Path, tuple[int, int], bool]:
    """Return the clip in the upright picture, its size, and whether it is temporary.

    ``orientation_hint_degrees`` is a clockwise rotation from the encoded buffer
    to the upright picture. Container display rotation is left off, so this hint
    is applied once. ``0`` keeps the original file. ``90``, ``180``, and ``270``
    (modulo 360) are written to a temporary video the caller must delete.
    """
    code = _clockwise_rotation_code(orientation_hint_degrees)
    if code is None:
        return video_path, _encoded_frame_size(video_path), False
    path, size = _write_rotated_video(video_path, code)
    return path, size, True


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
    temporary = False
    upright_path = video
    try:
        upright_path, upright_size, temporary = upright_video(
            video,
            sidecar.orientationHintDegrees,
        )
        _require_upright_size(upright_size, sidecar)
        track = track_video(upright_path, fps=fps, config=config, method=detection_method)
    finally:
        if temporary:
            upright_path.unlink(missing_ok=True)
    timestamps = sidecar.presentationTimestampsUs
    if _use_presentation_clock(sidecar, track):
        track = apply_presentation_times(track, timestamps)
    if len(track) < 5:
        raise NoBallFoundError(f"Insufficient track points ({len(track)}) in {video}")

    return _capture_result(sidecar, video, track, calib, config, fps)


def _capture_result(
    sidecar: CaptureSidecar,
    video: Path,
    track: list[TrackPoint],
    calib: CameraCalibration,
    config: dict,
    fps: float,
) -> dict:
    """Fit the track and build the capture result.

    ``checks`` holds plain-language warnings when the result looks wrong;
    ``warnings`` keeps the timing notes about the phone's clock.
    """
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
        "timing_source": _timing_source(sidecar, track),
        "warnings": _warnings(sidecar, track),
        "checks": result_checks(metrics_dict, fps=fps),
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


def _timestamps_cover_track(timestamps: list[int], track: list[TrackPoint]) -> bool:
    return bool(timestamps) and all(0 <= point.frame_idx < len(timestamps) for point in track)


def _durations_match(sidecar: CaptureSidecar) -> bool:
    wall = sidecar.wallClockSeconds
    duration = sidecar.durationSeconds
    if not _positive_finite(wall) or not _positive_finite(duration):
        return False
    wall_s = float(wall)
    duration_s = float(duration)
    return abs(wall_s - duration_s) / max(wall_s, duration_s) <= 0.05


def _use_presentation_clock(sidecar: CaptureSidecar, track: list[TrackPoint]) -> bool:
    covered = _timestamps_cover_track(sidecar.presentationTimestampsUs, track)
    return covered and _durations_match(sidecar)


def _timing_source(sidecar: CaptureSidecar, track: list[TrackPoint]) -> str:
    if _use_presentation_clock(sidecar, track):
        return "presentation_timestamps"
    return "frame_index"


def _warnings(sidecar: CaptureSidecar, track: list[TrackPoint]) -> list[str]:
    warnings: list[str] = []
    timestamps = sidecar.presentationTimestampsUs
    if not timestamps:
        warnings.append(_NO_TIMESTAMPS)
    elif not _timestamps_cover_track(timestamps, track):
        warnings.append(_PARTIAL_TIMESTAMPS)
    elif not _durations_match(sidecar):
        warnings.append(_CLOCK_MISMATCH)
    measured = sidecar.measuredFps
    capture = float(sidecar.captureFps)
    if (
        _positive_finite(measured)
        and _positive_finite(capture)
        and abs(float(measured) - capture) / capture > 0.05
    ):
        warnings.append(_FPS_MISMATCH)
    return warnings


def _clockwise_rotation_code(orientation_hint_degrees: int) -> int | None:
    angle = int(orientation_hint_degrees) % 360
    if angle == 0:
        return None
    if angle == 90:
        return cv2.ROTATE_90_CLOCKWISE
    if angle == 180:
        return cv2.ROTATE_180
    if angle == 270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    raise ValueError(
        f"orientation hint {orientation_hint_degrees} must be a multiple of 90 degrees"
    )


def _open_encoded_capture(video_path: Path) -> cv2.VideoCapture:
    """Open the encoded buffer with container display rotation left off."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoOpenError(f"Cannot open video: {video_path}")
    prop = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if prop is not None:
        capture.set(prop, 0)
    return capture


def _encoded_frame_size(video_path: Path) -> tuple[int, int]:
    capture = _open_encoded_capture(video_path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    return width, height


# Lossless first: the ball is a few pixels wide, and a lossy mp4v copy of a
# real clip shifted pixels by up to 4.7 grey levels on average and changed the
# frames the tracker found the ball in (docs/REAL_FOOTAGE.md). FFV1 costs disk
# (roughly 0.45 MB per 900x474 frame) but only for the life of the analysis.
_UPRIGHT_CODECS: tuple[tuple[str, str], ...] = (("FFV1", ".mkv"), ("mp4v", ".mp4"))


def _open_upright_writer(
    fps: float, size: tuple[int, int]
) -> tuple[cv2.VideoWriter, Path] | tuple[None, None]:
    """Open a temporary writer, lossless when this OpenCV build allows it."""
    for fourcc, suffix in _UPRIGHT_CODECS:
        descriptor, name = tempfile.mkstemp(suffix=suffix)
        os.close(descriptor)
        path = Path(name)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()
        path.unlink(missing_ok=True)
    return None, None


def _write_rotated_video(video_path: Path, rotate_code: int) -> tuple[Path, tuple[int, int]]:
    capture = _open_encoded_capture(video_path)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    writer_fps = source_fps if source_fps > 1.0 else 30.0
    temp_path: Path | None = None
    writer: cv2.VideoWriter | None = None
    try:
        ok, frame = capture.read()
        if not ok:
            raise VideoOpenError(f"Cannot read frames from {video_path}")
        upright = cv2.rotate(frame, rotate_code)
        height, width = upright.shape[:2]
        writer, temp_path = _open_upright_writer(writer_fps, (width, height))
        if writer is None or temp_path is None:
            raise RuntimeError("Failed to open an upright video writer")
        writer.write(upright)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(cv2.rotate(frame, rotate_code))
        return temp_path, (width, height)
    except Exception:
        if writer is not None:
            writer.release()
            writer = None
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def _require_upright_size(actual: tuple[int, int], sidecar: CaptureSidecar) -> None:
    expected = (int(sidecar.uprightWidth), int(sidecar.uprightHeight))
    if actual != expected:
        raise RuntimeError(
            f"Upright video is {actual[0]}x{actual[1]} "
            f"but the sidecar upright size is {expected[0]}x{expected[1]}"
        )
