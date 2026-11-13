"""Camera calibration from stump landmarks and projection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import load_config
from .errors import CalibrationFailedError, CalibrationInputError
from .pitch import stump_landmarks_3d
from .synthetic import CameraModel


@dataclass
class CameraCalibration:
    """Pinhole camera calibration (world frame matches pitch.py)."""

    K: np.ndarray  # 3x3 intrinsic matrix
    R: np.ndarray  # 3x3 world -> camera rotation
    t: np.ndarray  # (3,) world -> camera translation
    resolution: tuple[int, int]
    rms_reprojection_error: float

    @property
    def camera_center_world(self) -> np.ndarray:
        """Camera optical centre in world coordinates."""
        return -self.R.T @ self.t


def _intrinsic_from_fov(resolution: tuple[int, int], fov_deg: float) -> np.ndarray:
    w, h = resolution
    f = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def project_points(calib: CameraCalibration, points_3d: np.ndarray) -> np.ndarray:
    """Project Nx3 world points to Nx2 pixel coordinates."""
    pts = np.asarray(points_3d, dtype=np.float64)
    cam = (calib.R @ pts.T).T + calib.t
    depth = cam[:, 2]
    uv_h = (calib.K @ cam.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    uv[np.abs(depth) < 1e-9] = np.nan
    return uv


def backproject_ray(
    calib: CameraCalibration, uv: np.ndarray | tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (camera origin in world, unit ray direction in world) for pixel ``uv``."""
    uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    k_inv = np.linalg.inv(calib.K)
    ones = np.ones((len(uv_arr), 1), dtype=np.float64)
    hom = np.hstack([uv_arr, ones])
    dir_cam = (k_inv @ hom.T).T
    dir_cam /= np.linalg.norm(dir_cam, axis=1, keepdims=True)
    dir_world = (calib.R.T @ dir_cam.T).T
    origin = calib.camera_center_world
    if len(dir_world) == 1:
        return origin, dir_world[0]
    return origin, dir_world


def reprojection_rmse(
    calib: CameraCalibration,
    image_points: dict[str, np.ndarray],
    landmarks_3d: dict[str, np.ndarray] | None = None,
) -> float:
    """Root-mean-square reprojection error in pixels."""
    if landmarks_3d is None:
        landmarks_3d = stump_landmarks_3d()
    keys = [k for k in image_points if k in landmarks_3d]
    if not keys:
        return float("inf")
    pts_3d = np.stack([landmarks_3d[k] for k in keys])
    pts_2d = np.stack([np.asarray(image_points[k], dtype=np.float64)[:2] for k in keys])
    proj = project_points(calib, pts_3d)
    valid = np.isfinite(proj).all(axis=1)
    if not np.any(valid):
        return float("inf")
    err = proj[valid] - pts_2d[valid]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def is_calibration_acceptable(rmse_px: float, max_rmse: float = 5.0) -> bool:
    """Return True if reprojection RMSE is within tolerance."""
    return rmse_px <= max_rmse


def _solve_pose(
    pts_3d: np.ndarray, pts_2d: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run PnP (falling back to RANSAC); any OpenCV failure becomes a typed error."""
    try:
        ok, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                pts_3d,
                pts_2d,
                K,
                dist,
                iterationsCount=200,
                reprojectionError=8.0,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
    except cv2.error as exc:
        raise CalibrationFailedError(f"solvePnP rejected the landmark points: {exc}") from exc
    if not ok:
        raise CalibrationFailedError("solvePnP failed to estimate camera pose")
    return rvec, tvec


def solve_pnp(
    image_points: dict[str, np.ndarray],
    landmarks_3d: dict[str, np.ndarray] | None = None,
    image_size: tuple[int, int] | None = None,
    K: np.ndarray | None = None,
    fov_deg: float | None = None,
) -> CameraCalibration:
    """Estimate camera pose from 2D stump landmarks via PnP."""
    if landmarks_3d is None:
        landmarks_3d = stump_landmarks_3d()

    keys = sorted(k for k in image_points if k in landmarks_3d)
    if len(keys) < 4:
        raise CalibrationInputError(f"Need at least 4 landmark correspondences, got {len(keys)}")

    pts_2d = np.array([image_points[k][:2] for k in keys], dtype=np.float64)
    pts_3d = np.array([landmarks_3d[k] for k in keys], dtype=np.float64)

    if image_size is None:
        cfg = load_config()
        w, h = cfg["camera"]["resolution"]
        image_size = (int(w), int(h))

    if K is None:
        if fov_deg is None:
            fov_deg = float(load_config()["camera"]["fov_deg"])
        K = _intrinsic_from_fov(image_size, fov_deg)

    dist = np.zeros(5, dtype=np.float64)
    rvec, tvec = _solve_pose(pts_3d, pts_2d, K, dist)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    calib = CameraCalibration(
        K=K,
        R=R,
        t=t,
        resolution=image_size,
        rms_reprojection_error=0.0,
    )
    calib.rms_reprojection_error = reprojection_rmse(calib, image_points, landmarks_3d)
    return calib


def project_stumps_from_synthetic_camera(camera: CameraModel) -> CameraCalibration:
    """Build ``CameraCalibration`` from a synthetic ``CameraModel``."""
    K = camera.intrinsic_matrix()
    R, t = camera.extrinsic_matrix()
    landmarks = stump_landmarks_3d()
    image_points: dict[str, np.ndarray] = {}
    for key, pt in landmarks.items():
        uv = camera.project(pt.reshape(1, 3))[0]
        if np.all(np.isfinite(uv)):
            image_points[key] = uv
    calib = CameraCalibration(
        K=K,
        R=R,
        t=t,
        resolution=camera.resolution,
        rms_reprojection_error=0.0,
    )
    calib.rms_reprojection_error = reprojection_rmse(calib, image_points, landmarks)
    return calib


def calibration_from_camera_model(camera: CameraModel) -> CameraCalibration:
    """Alias for ``project_stumps_from_synthetic_camera``."""
    return project_stumps_from_synthetic_camera(camera)
