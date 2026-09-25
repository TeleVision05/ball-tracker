"""Save and load camera calibration JSON for real-video analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import CameraCalibration


def _calib_to_dict(
    calib: CameraCalibration,
    stump_image_points: dict | None = None,
) -> dict:
    data: dict = {
        "K": calib.K.tolist(),
        "R": calib.R.tolist(),
        "t": calib.t.reshape(-1).tolist(),
        "resolution": list(calib.resolution),
        "rms": float(calib.rms_reprojection_error),
    }
    if stump_image_points is not None:
        data["stump_image_points"] = {
            k: [float(v[0]), float(v[1])] for k, v in stump_image_points.items()
        }
    return data


def save_calibration(
    calib: CameraCalibration,
    path: str | Path,
    stump_image_points: dict | None = None,
) -> None:
    """Write calibration to JSON (K, R, t, resolution, rms, stump_image_points)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_calib_to_dict(calib, stump_image_points), f, indent=2)


def load_calibration(path: str | Path) -> tuple[CameraCalibration, dict]:
    """Load calibration JSON; return (CameraCalibration, stump_image_points dict)."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    resolution = tuple(int(x) for x in data["resolution"])
    calib = CameraCalibration(
        K=np.asarray(data["K"], dtype=np.float64),
        R=np.asarray(data["R"], dtype=np.float64),
        t=np.asarray(data["t"], dtype=np.float64).reshape(3),
        resolution=resolution,  # type: ignore[arg-type]
        rms_reprojection_error=float(data.get("rms", data.get("rms_reprojection_error", 0.0))),
    )
    stump_points: dict = {}
    if "stump_image_points" in data:
        stump_points = {
            k: np.asarray(v, dtype=np.float64) for k, v in data["stump_image_points"].items()
        }
    return calib, stump_points
