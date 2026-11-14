"""Merge and dedupe detections coming from overlapping sliding windows.

Because consecutive windows overlap by `window_overlap` frames, the same
global frame_index can be detected multiple times (once per window that
covered it). We keep exactly one detection per frame_index, preferring:
  1. A visible detection over a non-visible one.
  2. Among detections with the same visibility, the higher confidence one.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _is_better(candidate: Dict[str, Any], existing: Dict[str, Any]) -> bool:
    candidate_visible = bool(candidate.get("ball_visible"))
    existing_visible = bool(existing.get("ball_visible"))

    if candidate_visible != existing_visible:
        return candidate_visible  # prefer visible over not-visible

    return float(candidate.get("confidence") or 0.0) > float(existing.get("confidence") or 0.0)


def merge_detections(raw_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse raw per-window detections into one entry per frame_index.

    Each item in `raw_detections` is expected to have at least:
    frame_index, ball_visible, x, y, confidence, source_window.

    Returns a list sorted ascending by frame_index.
    """
    best_by_frame: Dict[int, Dict[str, Any]] = {}

    for detection in raw_detections:
        frame_index = detection["frame_index"]
        existing = best_by_frame.get(frame_index)
        if existing is None or _is_better(detection, existing):
            best_by_frame[frame_index] = detection

    return [best_by_frame[key] for key in sorted(best_by_frame)]
