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
from typing import Any, Dict, List, Optional

from app.config import (
    ANALYZE_FRAMES_INDIVIDUALLY,
    JPEG_QUALITY,
    OLLAMA_MAX_FRAME_WIDTH,
)
from app.models import database as db
from app.services import merge, progress
from app.services.ollama_client import OllamaClient, OllamaError
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

        video_path = analysis["video_path"]
        fps = analysis["fps"] or 30.0
        width = analysis["video_width"] or 0
        height = analysis["video_height"] or 0
        frame_count = analysis["frame_count"] or 0

        client = OllamaClient(base_url=base_url, model=model)
        all_raw: List[Dict[str, Any]] = []
        cancelled = False

        if ANALYZE_FRAMES_INDIVIDUALLY:
            # Fast path: each unique (strided) frame once — no overlap rework.
            frame_indices = sample_frame_indices(frame_count, frame_stride)
            total_jobs = len(frame_indices)
            progress.start(analysis_id, total_jobs)
            progress.append_log(
                analysis_id,
                f"analysis_id={analysis_id} frames={frame_count} jobs={total_jobs} "
                f"stride={frame_stride} mode=per-frame model={model}",
            )

            for job_index, frame_index in enumerate(frame_indices):
                if progress.is_cancelled(analysis_id):
                    cancelled = True
                    progress.append_log(analysis_id, f"Cancelled before frame {frame_index}")
                    break

                progress.update(analysis_id, current_window=job_index)
                progress.append_log(
                    analysis_id,
                    f"frame {job_index + 1}/{total_jobs} (index={frame_index}) → Ollama",
                )

                try:
                    frames_bgr = extract_frames_bgr(video_path, frame_index, frame_index + 1)
                    if not frames_bgr:
                        progress.append_log(analysis_id, f"frame={frame_index} missing, skipping")
                        progress.update(analysis_id, current_window=job_index + 1)
                        continue

                    encoded = encode_frame_for_ollama(
                        frames_bgr[0],
                        max_width=OLLAMA_MAX_FRAME_WIDTH,
                        quality=JPEG_QUALITY,
                    )
                    item = await client.detect_single_frame(_to_b64(encoded.jpeg_bytes))
                    x, y = item.x, item.y
                    if item.ball_visible and x is not None and y is not None:
                        x, y = encoded.scale_to_original(x, y)
                    all_raw.append(
                        {
                            "frame_index": frame_index,
                            "ball_visible": item.ball_visible,
                            "x": x,
                            "y": y,
                            "confidence": item.confidence,
                            "source_window": job_index,
                        }
                    )
                except OllamaError as exc:
                    logger.warning(
                        "analysis_id=%s frame=%s Ollama failed: %s",
                        analysis_id,
                        frame_index,
                        exc,
                    )
                    progress.append_log(
                        analysis_id, f"frame={frame_index} failed: {exc} (continuing)"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "analysis_id=%s frame=%s unexpected error", analysis_id, frame_index
                    )
                    progress.append_log(
                        analysis_id,
                        f"frame={frame_index} unexpected error: {exc} (continuing)",
                    )

                progress.update(
                    analysis_id,
                    current_window=job_index + 1,
                    detections_count=len(merge.merge_detections(all_raw)),
                )
        else:
            # Contact-sheet path: true temporal windows (one Ollama call each).
            windows = compute_windows(frame_count, window_size, window_overlap)
            total_windows = len(windows)
            progress.start(analysis_id, total_windows)
            progress.append_log(
                analysis_id,
                f"analysis_id={analysis_id} frames={frame_count} windows={total_windows} "
                f"window_size={window_size} overlap={window_overlap} "
                f"mode=contact-sheet model={model}",
            )

            for window_index, (start, end) in enumerate(windows):
                if progress.is_cancelled(analysis_id):
                    cancelled = True
                    progress.append_log(analysis_id, f"Cancelled before window {window_index}")
                    break

                progress.update(analysis_id, current_window=window_index)
                progress.append_log(
                    analysis_id,
                    f"window {window_index + 1}/{total_windows} frames=[{start},{end}) → Ollama",
                )

                try:
                    frames_bgr = extract_frames_bgr(video_path, start, end)
                    if not frames_bgr:
                        progress.append_log(
                            analysis_id,
                            f"window={window_index} produced no frames, skipping",
                        )
                        continue
                    # Apply stride inside the window for fewer panels.
                    step = max(1, frame_stride)
                    sampled = frames_bgr[::step]
                    window_raw = await _analyze_contact_sheet(
                        client=client,
                        frames_bgr=sampled,
                        start=start,
                        end=end,
                        window_index=window_index,
                        local_stride=step,
                    )
                    all_raw.extend(window_raw)
                except OllamaError as exc:
                    logger.warning(
                        "analysis_id=%s window=%s Ollama failed: %s",
                        analysis_id,
                        window_index,
                        exc,
                    )
                    progress.append_log(
                        analysis_id, f"window={window_index} failed: {exc} (continuing)"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "analysis_id=%s window=%s unexpected error",
                        analysis_id,
                        window_index,
                    )
                    progress.append_log(
                        analysis_id,
                        f"window={window_index} unexpected error: {exc} (continuing)",
                    )

                progress.update(
                    analysis_id,
                    current_window=window_index + 1,
                    detections_count=len(merge.merge_detections(all_raw)),
                )

        merged_raw = merge.merge_detections(all_raw)
        detection_records = [build_detection_record(raw, fps, width, height) for raw in merged_raw]

        await db.clear_detections(analysis_id)
        await db.insert_detections(analysis_id, detection_records)

        final_status = "cancelled" if cancelled else "completed"
        await db.update_analysis(analysis_id, status=final_status, error=None)
        progress.update(
            analysis_id,
            status=final_status,
            detections_count=len(detection_records),
        )
        progress.append_log(
            analysis_id,
            f"analysis_id={analysis_id} finished status={final_status} "
            f"detections={len(detection_records)} "
            f"elapsed={time.monotonic() - started_at:.1f}s",
        )

    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logger.exception("analysis_id=%s pipeline crashed", analysis_id)
        await db.update_analysis(analysis_id, status="failed", error=str(exc))
        progress.update(analysis_id, status="failed", message=str(exc))
        progress.append_log(analysis_id, f"Pipeline failed: {exc}")


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
