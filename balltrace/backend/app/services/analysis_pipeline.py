"""Sliding-window / strided analysis orchestration.

Ollama vision is the sole ball detector. For thinking VLMs (qwen3-vl), one
frame per request is much more reliable than multi-image chats — but that is
also slow. Optimizations:

* Analyze every Nth frame (``frame_stride``) — biggest wall-clock win.
* Never re-analyze the same frame just because windows overlap.
* Downscale frames before Ollama; scale coordinates back afterwards.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import (
    ANALYZE_FRAMES_INDIVIDUALLY,
    JPEG_QUALITY,
    OLLAMA_MAX_FRAME_WIDTH,
)
from app.errors import (
    OllamaError,
    OllamaModelMissingError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from app.friendly_errors import FriendlyError, describe
from app.models import database as db
from app.services import merge, progress
from app.services.ollama_client import OllamaClient
from app.services.video import (
    build_contact_sheet,
    compute_windows,
    encode_frame_for_ollama,
    extract_frames_bgr,
    frame_to_timestamp,
)

logger = logging.getLogger(__name__)


def to_normalized(x: Optional[float], y: Optional[float], width: int, height: int) -> tuple:
    """Compute normalized (0..1) coordinates. Returns (None, None) if the
    detection has no coordinates or the frame dimensions are unknown."""
    if x is None or y is None or not width or not height:
        return None, None
    return x / width, y / height


def build_detection_record(
    raw: Dict[str, Any], fps: float, width: int, height: int
) -> Dict[str, Any]:
    """Turn a merged raw detection dict into a full Detection record ready
    for persistence, computing timestamp and normalized coordinates."""
    x = raw.get("x")
    y = raw.get("y")
    normalized_x, normalized_y = to_normalized(x, y, width, height)
    return {
        "id": str(uuid.uuid4()),
        "frame_index": raw["frame_index"],
        "timestamp": frame_to_timestamp(raw["frame_index"], fps),
        "ball_visible": bool(raw.get("ball_visible")),
        "x": x,
        "y": y,
        "normalized_x": normalized_x,
        "normalized_y": normalized_y,
        "confidence": float(raw.get("confidence") or 0.0),
        "source_window": raw.get("source_window", 0),
    }


def sample_frame_indices(frame_count: int, frame_stride: int) -> List[int]:
    """Return unique frame indices to send to Ollama."""
    if frame_count <= 0:
        return []
    stride = max(1, frame_stride)
    indices = list(range(0, frame_count, stride))
    last = frame_count - 1
    if indices and indices[-1] != last:
        indices.append(last)
    return indices


# Stop early when Ollama is unreachable for this many frames in a row, instead
# of grinding through the whole video and reporting "completed" with nothing.
MAX_CONSECUTIVE_OLLAMA_FAILURES = 3


class _AbortAnalysis(Exception):
    """Stop the run now; ``friendly`` is what the user is told."""

    def __init__(self, friendly: FriendlyError):
        super().__init__(friendly.message)
        self.friendly = friendly


@dataclass
class _Run:
    """Everything one analysis run needs, shared by the per-frame and window loops."""

    analysis_id: str
    client: OllamaClient
    video_path: str
    fps: float
    width: int
    height: int
    frame_count: int
    model: str
    all_raw: List[Dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False
    consecutive_ollama_failures: int = 0


async def run_analysis(
    analysis_id: str,
    base_url: str,
    model: str,
    window_size: int,
    window_overlap: int,
    frame_stride: int = 3,
) -> None:
    """Background task: run the full pipeline for one analysis."""
    started_at = time.monotonic()

    try:
        analysis = await db.get_analysis(analysis_id)
        if analysis is None:
            logger.error("analysis_id=%s not found when starting pipeline", analysis_id)
            return

        run = _Run(
            analysis_id=analysis_id,
            client=OllamaClient(base_url=base_url, model=model),
            video_path=analysis["video_path"],
            fps=analysis["fps"] or 30.0,
            width=analysis["video_width"] or 0,
            height=analysis["video_height"] or 0,
            frame_count=analysis["frame_count"] or 0,
            model=model,
        )
        if ANALYZE_FRAMES_INDIVIDUALLY:
            await _run_per_frame(run, frame_stride)
        else:
            await _run_windows(run, window_size, window_overlap, frame_stride)
        await _finish(run, started_at)
    except _AbortAnalysis as stop:
        await _fail(analysis_id, stop.friendly)
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logger.exception("analysis_id=%s pipeline crashed", analysis_id)
        await _fail(analysis_id, describe(exc, "running the analysis", log=False))


async def _fail(analysis_id: str, friendly: FriendlyError) -> None:
    """Record a failed run with a message the user can act on."""
    await db.update_analysis(analysis_id, status="failed", error=friendly.message)
    progress.update(analysis_id, status="failed", message=friendly.message)
    progress.append_log(analysis_id, f"Analysis stopped: {friendly.message}")


async def _finish(run: _Run, started_at: float) -> None:
    merged_raw = merge.merge_detections(run.all_raw)
    records = [build_detection_record(raw, run.fps, run.width, run.height) for raw in merged_raw]

    await db.clear_detections(run.analysis_id)
    await db.insert_detections(run.analysis_id, records)

    final_status = "cancelled" if run.cancelled else "completed"
    await db.update_analysis(run.analysis_id, status=final_status, error=None)
    progress.update(run.analysis_id, status=final_status, detections_count=len(records))
    progress.append_log(
        run.analysis_id,
        f"analysis_id={run.analysis_id} finished status={final_status} "
        f"detections={len(records)} elapsed={time.monotonic() - started_at:.1f}s",
    )


def _stop_if_cancelled(run: _Run, text: str) -> bool:
    if not progress.is_cancelled(run.analysis_id):
        return False
    run.cancelled = True
    progress.append_log(run.analysis_id, text)
    return True


def _record_job_failure(run: _Run, what: str, exc: Exception) -> None:
    """Log a failed frame/window for the user, or abort the run if it cannot continue.

    The raw exception goes to the server log; the progress log gets plain language.
    """
    if isinstance(exc, OllamaError):
        friendly = describe(exc, f"analysis_id={run.analysis_id} {what}")
        if isinstance(exc, OllamaModelMissingError):
            raise _AbortAnalysis(friendly) from exc
        if isinstance(exc, OllamaUnavailableError) and not isinstance(exc, OllamaTimeoutError):
            run.consecutive_ollama_failures += 1
            if run.consecutive_ollama_failures >= MAX_CONSECUTIVE_OLLAMA_FAILURES:
                raise _AbortAnalysis(friendly) from exc
        else:
            run.consecutive_ollama_failures = 0
    else:
        friendly = describe(exc, f"analysis_id={run.analysis_id} {what} (unexpected)")
    progress.append_log(run.analysis_id, f"{what} failed: {friendly.message} (continuing)")


async def _run_per_frame(run: _Run, frame_stride: int) -> None:
    """Fast path: each unique (strided) frame once, no overlap rework."""
    frame_indices = sample_frame_indices(run.frame_count, frame_stride)
    total_jobs = len(frame_indices)
    progress.start(run.analysis_id, total_jobs)
    progress.append_log(
        run.analysis_id,
        f"analysis_id={run.analysis_id} frames={run.frame_count} jobs={total_jobs} "
        f"stride={frame_stride} mode=per-frame model={run.model}",
    )

    for job_index, frame_index in enumerate(frame_indices):
        if _stop_if_cancelled(run, f"Cancelled before frame {frame_index}"):
            break
        progress.update(run.analysis_id, current_window=job_index)
        progress.append_log(
            run.analysis_id, f"frame {job_index + 1}/{total_jobs} (index={frame_index}) → Ollama"
        )
        try:
            raw = await _detect_one_frame(run, frame_index, job_index)
            if raw is None:
                progress.append_log(run.analysis_id, f"frame={frame_index} missing, skipping")
                progress.update(run.analysis_id, current_window=job_index + 1)
                continue
            run.all_raw.append(raw)
            run.consecutive_ollama_failures = 0
        except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the run
            _record_job_failure(run, f"frame={frame_index}", exc)
        _update_progress(run, job_index + 1)


async def _detect_one_frame(
    run: _Run, frame_index: int, job_index: int
) -> Optional[Dict[str, Any]]:
    """Ask Ollama about one frame; ``None`` if the video has no such frame."""
    frames_bgr = extract_frames_bgr(run.video_path, frame_index, frame_index + 1)
    if not frames_bgr:
        return None
    encoded = encode_frame_for_ollama(
        frames_bgr[0], max_width=OLLAMA_MAX_FRAME_WIDTH, quality=JPEG_QUALITY
    )
    item = await run.client.detect_single_frame(_to_b64(encoded.jpeg_bytes))
    x, y = item.x, item.y
    if item.ball_visible and x is not None and y is not None:
        x, y = encoded.scale_to_original(x, y)
    return {
        "frame_index": frame_index,
        "ball_visible": item.ball_visible,
        "x": x,
        "y": y,
        "confidence": item.confidence,
        "source_window": job_index,
    }


async def _run_windows(run: _Run, window_size: int, window_overlap: int, frame_stride: int) -> None:
    """Contact-sheet path: true temporal windows (one Ollama call each)."""
    windows = compute_windows(run.frame_count, window_size, window_overlap)
    total_windows = len(windows)
    progress.start(run.analysis_id, total_windows)
    progress.append_log(
        run.analysis_id,
        f"analysis_id={run.analysis_id} frames={run.frame_count} windows={total_windows} "
        f"window_size={window_size} overlap={window_overlap} "
        f"mode=contact-sheet model={run.model}",
    )

    step = max(1, frame_stride)
    for window_index, (start, end) in enumerate(windows):
        if _stop_if_cancelled(run, f"Cancelled before window {window_index}"):
            break
        progress.update(run.analysis_id, current_window=window_index)
        progress.append_log(
            run.analysis_id,
            f"window {window_index + 1}/{total_windows} frames=[{start},{end}) → Ollama",
        )
        try:
            frames_bgr = extract_frames_bgr(run.video_path, start, end)
            if not frames_bgr:
                progress.append_log(
                    run.analysis_id, f"window={window_index} produced no frames, skipping"
                )
                continue
            # Apply stride inside the window for fewer panels.
            run.all_raw.extend(
                await _analyze_contact_sheet(
                    client=run.client,
                    frames_bgr=frames_bgr[::step],
                    start=start,
                    end=end,
                    window_index=window_index,
                    local_stride=step,
                )
            )
            run.consecutive_ollama_failures = 0
        except Exception as exc:  # noqa: BLE001 - one bad window must not stop the run
            _record_job_failure(run, f"window={window_index}", exc)
        _update_progress(run, window_index + 1)


def _update_progress(run: _Run, completed_jobs: int) -> None:
    progress.update(
        run.analysis_id,
        current_window=completed_jobs,
        detections_count=len(merge.merge_detections(run.all_raw)),
    )


def _to_b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


async def _analyze_contact_sheet(
    client: OllamaClient,
    frames_bgr: List,
    start: int,
    end: int,
    window_index: int,
    local_stride: int = 1,
) -> List[Dict[str, Any]]:
    """Single Ollama call with a labeled montage of the window."""
    sheet = build_contact_sheet(frames_bgr)
    raw_items = await client.detect_balls_from_contact_sheet(
        _to_b64(sheet.jpeg_bytes), frame_count=sheet.frame_count
    )
    results: List[Dict[str, Any]] = []
    for item in raw_items:
        global_frame_index = start + (item.frame_index * local_stride)
        if global_frame_index >= end or item.frame_index < 0:
            continue
        x, y = item.x, item.y
        if item.ball_visible and x is not None and y is not None:
            if x >= sheet.panel_width:
                x = x - (item.frame_index * sheet.panel_width)
            x, y = sheet.scale_to_original(x, y)
        results.append(
            {
                "frame_index": global_frame_index,
                "ball_visible": item.ball_visible,
                "x": x,
                "y": y,
                "confidence": item.confidence,
                "source_window": window_index,
            }
        )
    return results
