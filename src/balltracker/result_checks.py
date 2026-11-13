"""Sanity checks that say, in plain words, when a delivery result should not be trusted.

The fit always returns numbers, even when the track behind them is a shirt, a
net post or two frames of a ball. On real clips that produced results such as
"199 km/h, bounced 7 m wide of the stumps" and "29 km/h" with nothing to tell the
user they were wrong (docs/REAL_FOOTAGE.md). These checks run on the metrics
the pipeline already produces and return sentences a user can act on. They
never change the numbers.

The bounds are wide on purpose. They are not accuracy claims: they only catch
values no real delivery from the bowler's end can have.
"""

from __future__ import annotations

import math

# Adult bowling, slowest spin to fastest recorded pace, with margin.
RELEASE_SPEED_KMH_RANGE = (50.0, 170.0)
# ``pitch_length_m`` is the bounce distance from the batter's stumps. A bounce
# a little behind them (a full toss landing past the stumps) is allowed.
BOUNCE_FROM_BATTER_M_RANGE = (-2.0, 20.12)
# The pitch is 3.05 m wide; allow a little past its edge either side of middle.
MAX_BOUNCE_LINE_M = 1.6
# The capture protocol asks for 120 fps. Warn below 100, which leaves room for
# a phone that measures a little under its requested rate.
PROTOCOL_FPS = 120.0
MIN_TRUSTED_FPS = 100.0
# Same threshold as ``metrics.compute_metrics`` uses for ``trajectory_fit_ok``.
MAX_FIT_ERROR_PX = 15.0

POOR_FIT = (
    "The tracked points don't follow a real ball flight{detail}. Treat these numbers as "
    "unreliable. Trim the clip to just the delivery, or mark the ball by hand, and run it again."
)
POOR_FIT_DETAIL = " (the fitted path misses them by {rmse:.0f} pixels on average)"
AT_FIT_LIMIT = (
    "The fit ran into the edge of the range it searches, so the speed and bounce are less "
    "certain than usual. Check the replay to see whether the tracked points are on the ball."
)
SPEED_OUT_OF_RANGE = (
    "A release speed of {speed:.0f} km/h is outside what real bowling produces "
    "({low:.0f} to {high:.0f} km/h), so the tracker probably followed something other "
    "than the ball, or the clip's frame rate is wrong. Check the replay and the frame rate."
)
BOUNCE_OFF_PITCH = (
    "The bounce came out off the pitch, which a real delivery from this end can't do. "
    "The stump setup may not match this clip (for example, the camera moved), or the "
    "tracker lost the ball. Set up the stumps again on a frame from this clip."
)
LOW_FRAME_RATE = (
    "This clip plays at {fps:.0f} frames per second; the capture guide asks for {need:.0f} or "
    "more, so treat the speed as rough. A slow-motion export can also play at a lower rate than it "
    "was filmed at, which makes the speed wrong; use the original file from the camera."
)


def result_checks(metrics: dict, fps: float | None = None) -> list[str]:
    """Plain-language warnings for a metrics dict from ``metrics_to_dict``.

    Returns an empty list when nothing looks wrong. ``fps`` is the frame rate
    the times came from; it is checked only when given.
    """
    checks: list[str] = []
    checks.extend(_fit_checks(metrics))
    checks.extend(_value_checks(metrics))
    if fps is not None and _finite(fps) and float(fps) < MIN_TRUSTED_FPS:
        checks.append(LOW_FRAME_RATE.format(fps=float(fps), need=PROTOCOL_FPS))
    return checks


def _fit_checks(metrics: dict) -> list[str]:
    checks: list[str] = []
    confidence = metrics.get("confidence") or {}
    rmse = metrics.get("fit_error_px")
    if confidence.get("trajectory_fit_ok") is False:
        detail = POOR_FIT_DETAIL.format(rmse=float(rmse)) if _finite(rmse) else ""
        checks.append(POOR_FIT.format(detail=detail))
    if confidence.get("within_fit_limits") is False:
        checks.append(AT_FIT_LIMIT)
    return checks


def _value_checks(metrics: dict) -> list[str]:
    checks: list[str] = []
    speed = metrics.get("release_speed_kmh")
    low, high = RELEASE_SPEED_KMH_RANGE
    if _finite(speed) and not low <= float(speed) <= high:
        checks.append(SPEED_OUT_OF_RANGE.format(speed=float(speed), low=low, high=high))
    length = metrics.get("pitch_length_m")
    line = metrics.get("pitch_line_m")
    near, far = BOUNCE_FROM_BATTER_M_RANGE
    off_length = _finite(length) and not near <= float(length) <= far
    off_line = _finite(line) and abs(float(line)) > MAX_BOUNCE_LINE_M
    if off_length or off_line:
        checks.append(BOUNCE_OFF_PITCH)
    return checks


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
