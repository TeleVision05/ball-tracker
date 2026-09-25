"""Trajectory helpers: group visible detections into continuous segments,
breaking the line wherever the gap between consecutive visible frames is
too large to plausibly represent the same, continuously-tracked ball flight.
"""
from __future__ import annotations

from typing import Any, Dict, List

Point = Dict[str, Any]


def should_connect(frame_a: int, frame_b: int, max_gap_frames: int) -> bool:
    """Whether two detections (frame_a before frame_b) should be connected
    by a trajectory line, given the maximum allowed frame gap."""
    return (frame_b - frame_a) <= max_gap_frames


def build_trajectory_segments(
    detections: List[Point], max_gap_frames: int
) -> List[List[Point]]:
    """Group visible detections (sorted by frame_index) into segments.

    A new segment starts whenever the frame gap between consecutive visible
    detections exceeds `max_gap_frames`, so the trajectory is never drawn
    across a long occlusion/miss.
    """
    visible = sorted(
        (
            d
            for d in detections
            if d.get("ball_visible") and d.get("x") is not None and d.get("y") is not None
        ),
        key=lambda d: d["frame_index"],
    )

    segments: List[List[Point]] = []
    current: List[Point] = []
    prev: Point | None = None

    for point in visible:
        if prev is not None and not should_connect(
            prev["frame_index"], point["frame_index"], max_gap_frames
        ):
            if current:
                segments.append(current)
            current = []
        current.append(point)
        prev = point

    if current:
        segments.append(current)

    return segments
