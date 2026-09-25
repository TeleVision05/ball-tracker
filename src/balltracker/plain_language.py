"""Plain-language descriptions of delivery metrics for non-technical readers."""

from __future__ import annotations

# Length bands measured from the batter's stumps (``pitch_length_m``), tuned
# for a typical fast-medium delivery on a full-size pitch.
_YORKER_MAX_M = 2.0
_FULL_MAX_M = 4.0
_GOOD_MAX_M = 7.0
_SHORT_MAX_M = 10.0

# Line band half-width either side of middle stump.
_MIDDLE_HALF_WIDTH_M = 0.1


def length_label(pitch_length_m: float) -> str:
    """Classify delivery length from distance-to-batter-stumps (metres)."""
    length = float(pitch_length_m)
    if length < _YORKER_MAX_M:
        return "yorker"
    if length < _FULL_MAX_M:
        return "full"
    if length < _GOOD_MAX_M:
        return "good length"
    if length < _SHORT_MAX_M:
        return "short"
    return "too short"


def line_label(pitch_line_m: float) -> str:
    """Classify delivery line from lateral bounce offset (metres).

    Assumes ``pitch_line_m`` follows the pipeline's convention (positive x =
    off stump side for a right-handed batter viewed from behind the bowler's
    stumps).
    """
    x = float(pitch_line_m)
    if abs(x) < _MIDDLE_HALF_WIDTH_M:
        return "middle stump"
    return "off stump" if x > 0 else "leg stump"


def bounce_description(pitch_line_m: float, pitch_length_m: float) -> str:
    """One-line plain-language bounce description, e.g. 'Good length, toward off stump'."""
    length = length_label(pitch_length_m)
    line = line_label(pitch_line_m)
    length_sentence = length[0].upper() + length[1:]
    if line == "middle stump":
        return f"{length_sentence}, on the stumps"
    return f"{length_sentence}, toward {line}"


def summarize_delivery(metrics: dict) -> dict:
    """Build a plain-language summary from a delivery metrics dict.

    Accepts the dict shape produced by ``metrics_to_dict`` /
    ``compute_metrics`` (keys ``release_speed_kmh``, ``pitch_line_m``,
    ``pitch_length_m``); missing keys degrade gracefully.
    """
    speed_kmh = metrics.get("release_speed_kmh")
    if speed_kmh is None:
        speed_text = "Speed unavailable"
    else:
        speed_text = f"{float(speed_kmh):.1f} km/h"

    pitch_length_m = metrics.get("pitch_length_m")
    pitch_line_m = metrics.get("pitch_line_m")
    if pitch_length_m is None or pitch_line_m is None:
        bounce_text = "Bounce location unavailable"
        length = "unknown"
        line = "unknown"
    else:
        bounce_text = bounce_description(pitch_line_m, pitch_length_m)
        length = length_label(pitch_length_m)
        line = line_label(pitch_line_m)

    return {
        "speed_kmh": float(speed_kmh) if speed_kmh is not None else None,
        "speed_text": speed_text,
        "bounce_text": bounce_text,
        "length": length,
        "line": line,
    }
