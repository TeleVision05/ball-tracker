"""Pydantic models: both the public API contract models and the internal
models used to validate raw Ollama vision responses."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Internal: raw Ollama detection response validation
# ---------------------------------------------------------------------------


class RawDetectionItem(BaseModel):
    frame_index: int
    ball_visible: bool
    x: Optional[float] = None
    y: Optional[float] = None
    confidence: float = 0.0


class RawDetectionResponse(BaseModel):
    # Required (not defaulted) so a response missing the "detections" key
    # entirely fails validation and triggers the correction-prompt retry
    # rather than silently being treated as "zero detections".
    detections: List[RawDetectionItem]


# ---------------------------------------------------------------------------
# Public API contract models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    ollama_available: bool
    ollama_url: str
    ollama_model: str
    ffmpeg_available: bool


class LimitsResponse(BaseModel):
    max_upload_mb: int


class OllamaSettingsResponse(BaseModel):
    base_url: str
    model: str
    window_size: int
    window_overlap: int
    frame_stride: int
    available: bool
    error: Optional[str] = None


class OllamaSettingsUpdate(BaseModel):
    base_url: Optional[str] = None
    model: Optional[str] = None
    window_size: Optional[int] = None
    window_overlap: Optional[int] = None
    frame_stride: Optional[int] = None


class Detection(BaseModel):
    id: str
    frame_index: int
    timestamp: float
    ball_visible: bool
    x: Optional[float] = None
    y: Optional[float] = None
    normalized_x: Optional[float] = None
    normalized_y: Optional[float] = None
    confidence: float
    source_window: int


class AnalysisBase(BaseModel):
    id: str
    name: str
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    fps: Optional[float] = None
    duration: Optional[float] = None
    frame_count: Optional[int] = None
    created_at: str
    status: str
    model: Optional[str] = None
    error: Optional[str] = None


class Analysis(AnalysisBase):
    detections: List[Detection] = Field(default_factory=list)


class AnalysisSummary(AnalysisBase):
    detection_count: int = 0


class CreateAnalysisRequest(BaseModel):
    name: str


class StartAnalysisRequest(BaseModel):
    model: Optional[str] = None
    window_size: Optional[int] = None
    window_overlap: Optional[int] = None
    frame_stride: Optional[int] = None


class ProgressEvent(BaseModel):
    analysis_id: str
    status: str
    current_window: int
    total_windows: int
    detections_count: int
    elapsed_seconds: float
    eta_seconds: Optional[float] = None
    message: str
    log: List[str] = Field(default_factory=list)


class ExportVideoRequest(BaseModel):
    line_thickness: Optional[int] = None
    max_gap_frames: Optional[int] = None
    show_markers: Optional[bool] = None


class ExportVideoResponse(BaseModel):
    job_id: str
    status: str
    download_url: Optional[str] = None
