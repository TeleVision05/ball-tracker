"""Men's cricket pitch geometry and 3D stump landmarks."""

from __future__ import annotations

import numpy as np

PITCH_LENGTH_M = 20.12
STUMP_HEIGHT_M = 0.711
STUMP_WIDTH_M = 0.2286  # outer-edge span of three stumps (9 in)
STUMP_DIAMETER_M = 0.0349
STUMP_WIDTH_SPACING_M = STUMP_WIDTH_M - STUMP_DIAMETER_M  # outer stump centre-to-centre
STUMP_ADJACENT_SPACING_M = (STUMP_WIDTH_M - 3.0 * STUMP_DIAMETER_M) / 2.0 + STUMP_DIAMETER_M

BALL_DIAMETER_M = 0.0715
BALL_RADIUS_M = BALL_DIAMETER_M / 2.0


def _stump_x_offsets() -> tuple[float, float, float]:
    """Return X offsets (m) for left, middle, right stump centres at one end."""
    half_span = STUMP_WIDTH_M / 2.0 - STUMP_DIAMETER_M / 2.0
    return (-half_span, 0.0, half_span)


def stump_landmarks_3d() -> dict[str, np.ndarray]:
    """Return 3D stump landmark positions in the world frame.

    Origin: bowler-end middle stump base.
    Axes: +Y toward batter, +X off-side (bowler's right), +Z up.

    Returns
    -------
    dict[str, np.ndarray]
        Keys like ``bowler_left_base``, ``batter_middle_top``, each shape (3,).
    """
    left_x, _, right_x = _stump_x_offsets()
    xs = {"left": left_x, "middle": 0.0, "right": right_x}
    ends = {"bowler": 0.0, "batter": PITCH_LENGTH_M}
    landmarks: dict[str, np.ndarray] = {}

    for end_name, y in ends.items():
        for stump_name, x in xs.items():
            base = np.array([x, y, 0.0], dtype=np.float64)
            top = np.array([x, y, STUMP_HEIGHT_M], dtype=np.float64)
            landmarks[f"{end_name}_{stump_name}_base"] = base
            landmarks[f"{end_name}_{stump_name}_top"] = top

    return landmarks
