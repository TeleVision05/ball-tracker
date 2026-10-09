"""Parse an Android capture sidecar and calibrate the camera from stump marks."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from .calibration import CameraCalibration, reprojection_rmse
from .pitch import stump_landmarks_3d
from .tracking import TrackPoint

_INTRINSIC_ERROR = "the phone did not report an upright intrinsic matrix"

# Image left is the bowler's left (world -X). Near is the bowler's end.
_MARKER_LANDMARKS: tuple[tuple[str, str], ...] = (
    ("nearLeft", "bowler_left_base"),
    ("nearRight", "bowler_right_base"),
    ("farLeft", "batter_left_base"),
    ("farRight", "batter_right_base"),
)


@dataclass
class CaptureSidecar:
    """Version-2 capture sidecar used to calibrate a phone camera."""

    version: int
    id: str
    videoFile: str
    encodedWidth: int
    encodedHeight: int
    uprightWidth: int
    uprightHeight: int
    orientationHintDegrees: int
    measuredFps: float | None
    captureFps: int
    presentationTimestampsUs: list[int]
    nearLeft: tuple[float, float]
    nearRight: tuple[float, float]
    farLeft: tuple[float, float]
    farRight: tuple[float, float]
    intrinsicMatrixUpright: np.ndarray


def load_capture_sidecar(path: Path) -> CaptureSidecar:
    """Load a version-2 capture sidecar from a JSON file."""
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("capture sidecar JSON must be an object")
    return capture_sidecar_from_dict(data)


def capture_sidecar_from_dict(data: dict) -> CaptureSidecar:
    """Parse a version-2 capture sidecar object."""
    version = data.get("version")
    if version != 2:
        raise ValueError(f"unsupported capture sidecar version {version}")

    intrinsic = _intrinsic_matrix(data.get("camera"))
    encoded_width = int(data["encodedWidth"])
    encoded_height = int(data["encodedHeight"])
    orientation = data.get("orientationHintDegrees", 0)
    orientation_degrees = 0 if orientation is None else int(orientation)
    upright_width, upright_height = _upright_size(
        encoded_width,
        encoded_height,
        data.get("uprightWidth"),
        data.get("uprightHeight"),
        orientation_degrees,
    )
    measured = data.get("measuredFps")
    markers = data["markers"]
    return CaptureSidecar(
        version=int(version),
        id=str(data["id"]),
        videoFile=str(data["videoFile"]),
        encodedWidth=encoded_width,
        encodedHeight=encoded_height,
        uprightWidth=upright_width,
        uprightHeight=upright_height,
        orientationHintDegrees=orientation_degrees,
        measuredFps=None if measured is None else float(measured),
        captureFps=int(data["captureFps"]),
        presentationTimestampsUs=_timestamps(data.get("presentationTimestampsUs")),
        nearLeft=_marker_pair(markers, "nearLeft"),
        nearRight=_marker_pair(markers, "nearRight"),
        farLeft=_marker_pair(markers, "farLeft"),
        farRight=_marker_pair(markers, "farRight"),
        intrinsicMatrixUpright=intrinsic,
    )


def marker_pixels(sidecar: CaptureSidecar) -> dict[str, np.ndarray]:
    """Map the four stump-mark fractions to upright-image pixels."""
    fractions = {
        "nearLeft": sidecar.nearLeft,
        "nearRight": sidecar.nearRight,
        "farLeft": sidecar.farLeft,
        "farRight": sidecar.farRight,
    }
    width = float(sidecar.uprightWidth)
    height = float(sidecar.uprightHeight)
    pixels: dict[str, np.ndarray] = {}
    for marker_name, landmark_name in _MARKER_LANDMARKS:
        fraction_x, fraction_y = fractions[marker_name]
        pixels[landmark_name] = np.array(
            [fraction_x * width, fraction_y * height],
            dtype=np.float64,
        )
    return pixels


def calibrate_from_capture(sidecar: CaptureSidecar) -> CameraCalibration:
    """Calibrate the camera from the four coplanar stump bases.

    ``cv2.SOLVEPNP_IPPE`` returns up to two planar poses. The pose whose
    camera centre is behind the bowler and above the pitch
    (world ``Y < -0.5`` and ``0.4 < Z < 4``) is kept. When both qualify,
    the lower reprojection RMSE wins. When neither does, raise.
    """
    image_points = marker_pixels(sidecar)
    landmarks = stump_landmarks_3d()
    keys = [landmark for _, landmark in _MARKER_LANDMARKS]
    object_points = np.ascontiguousarray(
        [landmarks[key] for key in keys],
        dtype=np.float64,
    )
    pixel_points = np.ascontiguousarray(
        [image_points[key] for key in keys],
        dtype=np.float64,
    )
    intrinsic = _usable_intrinsic(sidecar.intrinsicMatrixUpright)
    distortion = np.zeros(5, dtype=np.float64)
    _count, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
        object_points,
        pixel_points,
        intrinsic,
        distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )

    resolution = (int(sidecar.uprightWidth), int(sidecar.uprightHeight))
    calibrations: list[CameraCalibration] = []
    pose_rvecs = [] if rvecs is None else rvecs
    pose_tvecs = [] if tvecs is None else tvecs
    for rvec, tvec in zip(pose_rvecs, pose_tvecs):
        rvec_arr = np.asarray(rvec, dtype=np.float64)
        tvec_arr = np.asarray(tvec, dtype=np.float64)
        if not np.isfinite(rvec_arr).all() or not np.isfinite(tvec_arr).all():
            continue
        rotation, _ = cv2.Rodrigues(rvec_arr)
        translation = tvec_arr.reshape(3)
        calibrations.append(
            _calibration(
                intrinsic,
                rotation,
                translation,
                resolution,
                image_points,
                landmarks,
            )
        )
    if not calibrations:
        raise RuntimeError("solvePnPGeneric returned no usable camera pose")

    above_and_behind = [calib for calib in calibrations if _behind_bowler_and_above(calib)]
    if not above_and_behind:
        raise RuntimeError("no capture pose is behind the bowler and above the pitch")
    return min(above_and_behind, key=lambda calib: calib.rms_reprojection_error)


def presentation_times_s(timestamps_us: list[int]) -> list[float]:
    """Convert container presentation timestamps to seconds from the first frame."""
    if not timestamps_us:
        return []
    origin = timestamps_us[0]
    return [(int(timestamp) - origin) / 1_000_000 for timestamp in timestamps_us]


def apply_presentation_times(
    track: list[TrackPoint],
    timestamps_us: list[int],
) -> list[TrackPoint]:
    """Copy ``track`` with ``t_s`` taken from presentation timestamps when in range."""
    times = presentation_times_s(timestamps_us)
    updated: list[TrackPoint] = []
    for point in track:
        if 0 <= point.frame_idx < len(times):
            updated.append(replace(point, t_s=times[point.frame_idx]))
        else:
            updated.append(replace(point))
    return updated


def _upright_size(
    encoded_width: int,
    encoded_height: int,
    upright_width: object,
    upright_height: object,
    orientation_degrees: int,
) -> tuple[int, int]:
    if upright_width is None or upright_height is None:
        if orientation_degrees in (90, 270):
            return encoded_height, encoded_width
        return encoded_width, encoded_height
    return int(upright_width), int(upright_height)


def _timestamps(value: object) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    return [int(timestamp) for timestamp in value]


def _marker_pair(markers: dict, name: str) -> tuple[float, float]:
    pair = markers[name]
    return (float(pair[0]), float(pair[1]))


def _intrinsic_matrix(camera: object) -> np.ndarray:
    matrix = camera.get("intrinsicMatrixUpright") if isinstance(camera, dict) else None
    if matrix is None:
        raise ValueError(_INTRINSIC_ERROR)
    return _usable_intrinsic(matrix)


def _usable_intrinsic(matrix: object) -> np.ndarray:
    try:
        intrinsic = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(_INTRINSIC_ERROR) from exc
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError(_INTRINSIC_ERROR)
    if float(intrinsic[0, 0]) <= 0.0 or float(intrinsic[1, 1]) <= 0.0:
        raise ValueError(_INTRINSIC_ERROR)
    return intrinsic.copy()


def _behind_bowler_and_above(calib: CameraCalibration) -> bool:
    center = calib.camera_center_world
    return bool(center[1] < -0.5 and 0.4 < center[2] < 4.0)


def _calibration(
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    resolution: tuple[int, int],
    image_points: dict[str, np.ndarray],
    landmarks: dict[str, np.ndarray],
) -> CameraCalibration:
    calib = CameraCalibration(
        K=intrinsic.copy(),
        R=np.array(rotation, dtype=np.float64, copy=True),
        t=np.array(translation, dtype=np.float64, copy=True),
        resolution=resolution,
        rms_reprojection_error=0.0,
    )
    calib.rms_reprojection_error = reprojection_rmse(calib, image_points, landmarks)
    return calib
