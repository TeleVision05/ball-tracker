"""In-memory progress tracking for running analyses, consumed by the SSE
progress endpoint. State is intentionally not persisted -- it only needs to
survive for the lifetime of a running background task within this process.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional

_MAX_LOG_LINES = 200


@dataclass
class ProgressState:
    analysis_id: str
    status: str = "analyzing"
    current_window: int = 0
    total_windows: int = 0
    detections_count: int = 0
    message: str = ""
    log: List[str] = field(default_factory=list)
    cancel_requested: bool = False
    start_time: float = field(default_factory=time.monotonic)


_progress: Dict[str, ProgressState] = {}
_lock = Lock()


def start(analysis_id: str, total_windows: int) -> None:
    with _lock:
        _progress[analysis_id] = ProgressState(
            analysis_id=analysis_id,
            total_windows=total_windows,
            message="Starting analysis",
        )
        _progress[analysis_id].log.append("Starting analysis")


def update(analysis_id: str, **fields) -> None:
    with _lock:
        state = _progress.get(analysis_id)
        if state is None:
            return
        for key, value in fields.items():
            setattr(state, key, value)


def append_log(analysis_id: str, message: str) -> None:
    with _lock:
        state = _progress.get(analysis_id)
        if state is None:
            return
        state.log.append(message)
        if len(state.log) > _MAX_LOG_LINES:
            state.log = state.log[-_MAX_LOG_LINES:]
        state.message = message


def request_cancel(analysis_id: str) -> bool:
    with _lock:
        state = _progress.get(analysis_id)
        if state is None:
            return False
        state.cancel_requested = True
        return True


def is_cancelled(analysis_id: str) -> bool:
    with _lock:
        state = _progress.get(analysis_id)
        return bool(state and state.cancel_requested)


def is_active(analysis_id: str) -> bool:
    with _lock:
        state = _progress.get(analysis_id)
        return state is not None and state.status == "analyzing"


def snapshot(analysis_id: str) -> Optional[dict]:
    with _lock:
        state = _progress.get(analysis_id)
        if state is None:
            return None

        elapsed = time.monotonic() - state.start_time
        eta: Optional[float] = None
        if state.current_window > 0 and state.total_windows > 0 and state.status == "analyzing":
            per_window = elapsed / state.current_window
            remaining = max(state.total_windows - state.current_window, 0)
            eta = round(per_window * remaining, 2)

        return {
            "analysis_id": state.analysis_id,
            "status": state.status,
            "current_window": state.current_window,
            "total_windows": state.total_windows,
            "detections_count": state.detections_count,
            "elapsed_seconds": round(elapsed, 2),
            "eta_seconds": eta,
            "message": state.message,
            "log": list(state.log),
        }


def clear(analysis_id: str) -> None:
    with _lock:
        _progress.pop(analysis_id, None)
