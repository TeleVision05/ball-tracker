"""Shared stump landmark order and calibration helpers for CLI/UI."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .calib_io import save_calibration
from .calibration import CameraCalibration, reprojection_rmse, solve_pnp
from .errors import CalibrationInputError

LANDMARK_ORDER = [
    "bowler_left_base",
    "bowler_middle_base",
    "bowler_right_base",
    "bowler_left_top",
    "bowler_middle_top",
    "bowler_right_top",
    "batter_left_base",
    "batter_middle_base",
    "batter_right_base",
    "batter_left_top",
    "batter_middle_top",
    "batter_right_top",
]

LANDMARK_PLAIN_LABELS: dict[str, str] = {
    "bowler_left_base": "Bowler end — left stump base",
    "bowler_middle_base": "Bowler end — middle stump base",
    "bowler_right_base": "Bowler end — right stump base",
    "bowler_left_top": "Bowler end — left stump tip",
    "bowler_middle_top": "Bowler end — middle stump tip",
    "bowler_right_top": "Bowler end — right stump tip",
    "batter_left_base": "Batter end — left stump base",
    "batter_middle_base": "Batter end — middle stump base",
    "batter_right_base": "Batter end — right stump base",
    "batter_left_top": "Batter end — left stump tip",
    "batter_middle_top": "Batter end — middle stump tip",
    "batter_right_top": "Batter end — right stump tip",
}

WARN_RMSE_PX = 8.0
ABORT_RMSE_PX = 15.0
MIN_LANDMARKS = 6


def next_landmark_name(points: dict) -> str | None:
    """Return the next landmark name to click, or None if complete."""
    for name in LANDMARK_ORDER:
        if name not in points:
            return name
    return None


def calibrate_from_points(
    image_points: dict[str, np.ndarray],
    image_size: tuple[int, int],
    fov_deg: float = 55.0,
    out_path: str | Path | None = None,
) -> tuple[CameraCalibration, float]:
    """Solve PnP from stump clicks; optionally save JSON.

    Raises
    ------
    ValueError
        If fewer than ``MIN_LANDMARKS`` points or RMSE exceeds abort threshold.
    """
    if len(image_points) < MIN_LANDMARKS:
        raise CalibrationInputError(
            f"Need at least {MIN_LANDMARKS} landmarks, got {len(image_points)}"
        )

    pts = {k: np.asarray(v, dtype=np.float64) for k, v in image_points.items()}
    calib = solve_pnp(pts, image_size=image_size, fov_deg=fov_deg)
    rmse = reprojection_rmse(calib, pts)
    calib.rms_reprojection_error = rmse

    if rmse > ABORT_RMSE_PX:
        raise CalibrationInputError(
            f"Calibration RMSE {rmse:.2f} px exceeds {ABORT_RMSE_PX:.0f} px limit. "
            "Re-click stump landmarks more carefully."
        )

    if out_path is not None:
        save_calibration(calib, out_path, stump_image_points=pts)

    return calib, rmse
