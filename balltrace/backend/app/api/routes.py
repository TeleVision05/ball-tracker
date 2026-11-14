"""All BallTrace API endpoints, implementing balltrace/API_CONTRACT.md."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from app import config
from app.config import (
    PROGRESS_POLL_INTERVAL_SECONDS,
    analysis_dir,
    analysis_export_dir,
    ensure_directories,
)
from app.friendly_errors import app_error, describe
from app.models import database as db
from app.models.schemas import (
    Analysis,
    AnalysisSummary,
    CreateAnalysisRequest,
    Detection,
    ExportVideoRequest,
    HealthResponse,
    LimitsResponse,
    OllamaSettingsResponse,
    OllamaSettingsUpdate,
    StartAnalysisRequest,
)
from app.services import export_service, progress
from app.services.analysis_pipeline import run_analysis
from app.services.ollama_client import OllamaClient
from app.services.video import VideoError, get_video_metadata

logger = logging.getLogger(__name__)

router = APIRouter()

_UPLOAD_CHUNK_BYTES = 1024 * 1024

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

# Keep a reference to background analysis tasks so they aren't garbage
# collected mid-flight and so cancel can be more forceful if needed.
_running_tasks: Dict[str, asyncio.Task] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _video_url(analysis_id: str) -> str:
    return f"/api/analyses/{analysis_id}/video"


def _row_to_analysis(row: Dict[str, Any], detections: Optional[list] = None) -> Analysis:
    return Analysis(
        id=row["id"],
        name=row["name"],
        video_path=row["video_path"],
        video_url=_video_url(row["id"]) if row["video_path"] else None,
        video_width=row["video_width"],
        video_height=row["video_height"],
        fps=row["fps"],
        duration=row["duration"],
        frame_count=row["frame_count"],
        created_at=row["created_at"],
        status=row["status"],
        model=row["model"],
        error=row["error"],
        detections=[Detection(**d) for d in (detections or [])],
    )


def _row_to_summary(row: Dict[str, Any], detection_count: int) -> AnalysisSummary:
    return AnalysisSummary(
        id=row["id"],
        name=row["name"],
        video_path=row["video_path"],
        video_url=_video_url(row["id"]) if row["video_path"] else None,
        video_width=row["video_width"],
        video_height=row["video_height"],
        fps=row["fps"],
        duration=row["duration"],
        frame_count=row["frame_count"],
        created_at=row["created_at"],
        status=row["status"],
        model=row["model"],
        error=row["error"],
        detection_count=detection_count,
    )


async def _get_analysis_or_404(analysis_id: str) -> Dict[str, Any]:
    row = await db.get_analysis(analysis_id)
    if row is None:
        raise app_error("analysis_not_found")
    return row


# ---------------------------------------------------------------------------
# Health & settings
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = await db.get_settings()
    client = OllamaClient(base_url=settings["base_url"], model=settings["model"])
    available, _ = await client.check_available()
    return HealthResponse(
        status="ok",
        ollama_available=available,
        ollama_url=settings["base_url"],
        ollama_model=settings["model"],
        ffmpeg_available=export_service.ffmpeg_available(),
    )


@router.get("/limits", response_model=LimitsResponse)
async def limits() -> LimitsResponse:
    return LimitsResponse(max_upload_mb=config.MAX_UPLOAD_MB)


@router.get("/settings/ollama", response_model=OllamaSettingsResponse)
async def get_ollama_settings() -> OllamaSettingsResponse:
    settings = await db.get_settings()
    client = OllamaClient(base_url=settings["base_url"], model=settings["model"])
    available, error = await client.check_available()
    return OllamaSettingsResponse(**settings, available=available, error=error)


@router.put("/settings/ollama", response_model=OllamaSettingsResponse)
async def update_ollama_settings(payload: OllamaSettingsUpdate) -> OllamaSettingsResponse:
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    settings = await db.update_settings(updates)
    client = OllamaClient(base_url=settings["base_url"], model=settings["model"])
    available, error = await client.check_available()
    return OllamaSettingsResponse(**settings, available=available, error=error)


# ---------------------------------------------------------------------------
# Analyses: lifecycle
# ---------------------------------------------------------------------------


@router.post("/analyses", response_model=Analysis)
async def create_analysis(payload: CreateAnalysisRequest) -> Analysis:
    ensure_directories()
    analysis_id = str(uuid.uuid4())
    record = {
        "id": analysis_id,
        "name": payload.name,
        "video_path": None,
        "video_width": None,
        "video_height": None,
        "fps": None,
        "duration": None,
        "frame_count": None,
        "created_at": _now_iso(),
        "status": "created",
        "model": None,
        "window_size": None,
        "window_overlap": None,
        "error": None,
    }
    await db.create_analysis(record)
    row = await db.get_analysis(analysis_id)
    return _row_to_analysis(row)


def _save_upload_within_limit(file: UploadFile, dest_path: Path) -> None:
    """Copy the upload to ``dest_path``, refusing (and removing) anything over the size limit."""
    limit = config.max_upload_bytes()
    written = 0
    try:
        with dest_path.open("wb") as out_file:
            while chunk := file.file.read(_UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > limit:
                    raise app_error("upload_too_large", limit_mb=config.MAX_UPLOAD_MB)
                out_file.write(chunk)
    except BaseException:
        dest_path.unlink(missing_ok=True)
        raise


def _receive_and_validate_upload(file: UploadFile, final_path: Path, analysis_id: str):
    """Write the upload to a temporary file, check it, then move it into place.

    Anything that goes wrong leaves the analysis's existing video untouched and
    no temporary file behind. Returns the video metadata.
    """
    temp_path = final_path.with_name(f".incoming-{uuid.uuid4().hex}{final_path.suffix}")
    try:
        _save_upload_within_limit(file, temp_path)
        try:
            meta = get_video_metadata(temp_path)
        except VideoError as exc:
            raise describe(
                exc, f"reading the uploaded video for analysis_id={analysis_id}"
            ).to_app_error() from exc
        os.replace(temp_path, final_path)
        return meta
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


@router.post("/analyses/{analysis_id}/upload", response_model=Analysis)
async def upload_video(analysis_id: str, file: UploadFile) -> Analysis:
    row = await _get_analysis_or_404(analysis_id)

    target_dir = analysis_dir(analysis_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest_path = target_dir / f"source{suffix}"

    try:
        meta = _receive_and_validate_upload(file, dest_path, analysis_id)
    finally:
        _remove_folder_if_empty(target_dir)

    await db.update_analysis(
        analysis_id,
        video_path=str(dest_path),
        video_width=meta["width"],
        video_height=meta["height"],
        fps=meta["fps"],
        duration=meta["duration"],
        frame_count=meta["frame_count"],
        status="uploaded",
        error=None,
    )
    _remove_replaced_video(row.get("video_path"), dest_path)
    row = await db.get_analysis(analysis_id)
    return _row_to_analysis(row)


def _remove_folder_if_empty(folder: Path) -> None:
    try:
        folder.rmdir()
    except OSError:
        pass


def _remove_replaced_video(old_path: Optional[str], new_path: Path) -> None:
    """Delete the previous copy once the record points at the new one (different extension)."""
    if old_path and Path(old_path) != new_path:
        Path(old_path).unlink(missing_ok=True)


@router.post("/analyses/{analysis_id}/start")
async def start_analysis(analysis_id: str, payload: StartAnalysisRequest = StartAnalysisRequest()):
    row = await _get_analysis_or_404(analysis_id)
    if not row["video_path"]:
        raise app_error("no_video")

    settings = await db.get_settings()
    model = payload.model or settings["model"]
    window_size = payload.window_size or settings["window_size"]
    window_overlap = (
        payload.window_overlap if payload.window_overlap is not None else settings["window_overlap"]
    )
    frame_stride = (
        payload.frame_stride
        if payload.frame_stride is not None
        else settings.get("frame_stride", 3)
    )
    frame_stride = max(1, int(frame_stride))

    await db.update_analysis(
        analysis_id,
        status="analyzing",
        model=model,
        window_size=window_size,
        window_overlap=window_overlap,
        error=None,
    )

    task = asyncio.create_task(
        run_analysis(
            analysis_id=analysis_id,
            base_url=settings["base_url"],
            model=model,
            window_size=window_size,
            window_overlap=window_overlap,
            frame_stride=frame_stride,
        )
    )
    _running_tasks[analysis_id] = task
    task.add_done_callback(lambda _t: _running_tasks.pop(analysis_id, None))

    return {"status": "analyzing"}


@router.post("/analyses/{analysis_id}/cancel")
async def cancel_analysis(analysis_id: str):
    await _get_analysis_or_404(analysis_id)
    requested = progress.request_cancel(analysis_id)
    if not requested:
        # Not actively tracked in-memory (already finished, or this process
        # was restarted); nothing to cancel.
        return {"status": "not_running"}
    return {"status": "cancelling"}


# ---------------------------------------------------------------------------
# Analyses: retrieval
# ---------------------------------------------------------------------------


@router.get("/analyses")
async def list_analyses():
    rows = await db.list_analyses()
    summaries = []
    for row in rows:
        count = await db.count_detections(row["id"])
        summaries.append(_row_to_summary(row, count))
    return summaries


@router.get("/analyses/{analysis_id}", response_model=Analysis)
async def get_analysis(analysis_id: str) -> Analysis:
    row = await _get_analysis_or_404(analysis_id)
    detections = await db.get_detections(analysis_id)
    return _row_to_analysis(row, detections)


@router.get("/analyses/{analysis_id}/detections")
async def get_detections(analysis_id: str):
    await _get_analysis_or_404(analysis_id)
    detections = await db.get_detections(analysis_id)
    return [Detection(**d) for d in detections]


@router.get("/analyses/{analysis_id}/progress")
async def stream_progress(analysis_id: str):
    await _get_analysis_or_404(analysis_id)

    async def event_generator():
        while True:
            state = progress.snapshot(analysis_id)
            if state is None:
                current = await db.get_analysis(analysis_id)
                status = current["status"] if current else "unknown"
                count = await db.count_detections(analysis_id)
                state = {
                    "analysis_id": analysis_id,
                    "status": status,
                    "current_window": 0,
                    "total_windows": 0,
                    "detections_count": count,
                    "elapsed_seconds": 0.0,
                    "eta_seconds": None,
                    "message": f"Analysis is {status}",
                    "log": [],
                }
                yield f"data: {json.dumps(state)}\n\n"
                return

            yield f"data: {json.dumps(state)}\n\n"
            if state["status"] in _TERMINAL_STATUSES:
                return
            await asyncio.sleep(PROGRESS_POLL_INTERVAL_SECONDS)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/analyses/{analysis_id}/video")
async def get_video(analysis_id: str):
    row = await _get_analysis_or_404(analysis_id)
    if not row["video_path"]:
        raise app_error("video_missing")
    path = Path(row["video_path"])
    if not path.exists():
        raise app_error("video_missing")
    media_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
    return FileResponse(path, media_type=media_type, filename=path.name)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@router.get("/analyses/{analysis_id}/export/json")
async def export_json(analysis_id: str):
    row = await _get_analysis_or_404(analysis_id)
    detections = await db.get_detections(analysis_id)
    analysis_payload = _row_to_analysis(row, detections).model_dump()
    content = export_service.build_json_export(analysis_payload, detections)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{analysis_id}.json"'},
    )


@router.get("/analyses/{analysis_id}/export/csv")
async def export_csv(analysis_id: str):
    await _get_analysis_or_404(analysis_id)
    detections = await db.get_detections(analysis_id)
    content = export_service.build_csv_export(detections)
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{analysis_id}.csv"'},
    )


@router.post("/analyses/{analysis_id}/export/video")
async def export_video(analysis_id: str, payload: ExportVideoRequest = ExportVideoRequest()):
    row = await _get_analysis_or_404(analysis_id)
    if not row["video_path"]:
        raise app_error("no_video")

    detections = await db.get_detections(analysis_id)
    output_dir = analysis_export_dir(analysis_id)
    output_path = output_dir / "trajectory.mp4"

    line_thickness = payload.line_thickness or 3
    max_gap_frames = payload.max_gap_frames if payload.max_gap_frames is not None else 10
    show_markers = payload.show_markers if payload.show_markers is not None else True

    try:
        export_service.export_trajectory_video(
            video_path=row["video_path"],
            output_path=output_path,
            detections=detections,
            width=row["video_width"] or 0,
            height=row["video_height"] or 0,
            fps=row["fps"] or 30.0,
            line_thickness=line_thickness,
            max_gap_frames=max_gap_frames,
            show_markers=show_markers,
        )
    except (export_service.FFmpegNotAvailableError, VideoError) as exc:
        raise describe(
            exc, f"exporting the video for analysis_id={analysis_id}"
        ).to_app_error() from exc

    return {
        "job_id": str(uuid.uuid4()),
        "status": "completed",
        "download_url": f"/api/analyses/{analysis_id}/export/video/file",
    }


@router.get("/analyses/{analysis_id}/export/video/file")
async def download_exported_video(analysis_id: str):
    await _get_analysis_or_404(analysis_id)
    path = analysis_export_dir(analysis_id) / "trajectory.mp4"
    if not path.exists():
        raise app_error("export_missing")
    return FileResponse(path, media_type="video/mp4", filename=f"{analysis_id}_trajectory.mp4")
