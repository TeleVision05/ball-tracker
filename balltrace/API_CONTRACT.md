# BallTrace API Contract

Base URL: `http://localhost:8000`

## Health & Settings

### GET /api/health
```json
{
  "status": "ok",
  "ollama_available": true,
  "ollama_url": "http://localhost:11434",
  "ollama_model": "qwen3-vl:8b",
  "ffmpeg_available": true
}
```

### GET /api/limits
```json
{ "max_upload_mb": 500 }
```
The largest video `POST /api/analyses/{id}/upload` accepts. Set with `BALLTRACE_MAX_UPLOAD_MB`.

### GET /api/settings/ollama
```json
{
  "base_url": "http://localhost:11434",
  "model": "qwen3-vl:8b",
  "window_size": 4,
  "window_overlap": 0,
  "frame_stride": 3,
  "available": true,
  "error": null
}
```

### PUT /api/settings/ollama
Body: `{ "base_url"?: string, "model"?: string, "window_size"?: number, "window_overlap"?: number, "frame_stride"?: number }`
Returns updated settings object.

#### Defaults

These are the values the backend uses until they are changed in Settings. They
come from `backend/app/config.py`, and `backend/tests/test_api_contract_defaults.py`
fails if this table and that file disagree.

| Field | Default | Meaning |
| --- | --- | --- |
| `base_url` | `http://localhost:11434` | Ollama server |
| `model` | `qwen3-vl:8b` | Vision model |
| `window_size` | `4` | Frames per Ollama request in windowed mode |
| `window_overlap` | `0` | Frames shared between neighbouring windows in windowed mode |
| `frame_stride` | `3` | Analyze every Nth frame (`1` = every frame) |

The backend sets `ANALYZE_FRAMES_INDIVIDUALLY = True`. In that mode each sampled
frame is sent on its own, chosen by `frame_stride`, and `window_size` and
`window_overlap` are not used. They apply only if that constant is changed to
`False`.

## Analyses

### POST /api/analyses
Body: `{ "name": string }`
Returns: `Analysis` (status=`created`)

### POST /api/analyses/{id}/upload
multipart/form-data: `file` (video)
Returns: `Analysis` with video metadata (status=`uploaded`)
Errors: `413 upload_too_large` when the file is over the limit, `400 video_unreadable` when it is not a readable video.

### POST /api/analyses/{id}/start
Body (optional): `{ "model"?: string, "window_size"?: number, "window_overlap"?: number, "frame_stride"?: number }`
Omitted fields fall back to the saved settings (see Defaults above).
Starts background analysis. Returns `{ "status": "analyzing" }`

### POST /api/analyses/{id}/cancel
Cancels running analysis.

### GET /api/analyses
Returns: `AnalysisSummary[]` (no detections payload)

### GET /api/analyses/{id}
Returns: full `Analysis` including detections

### GET /api/analyses/{id}/detections
Returns: `Detection[]`

### GET /api/analyses/{id}/progress
SSE stream of progress events:
```json
{
  "analysis_id": "...",
  "status": "analyzing|completed|failed|cancelled",
  "current_window": 12,
  "total_windows": 84,
  "detections_count": 47,
  "elapsed_seconds": 42.1,
  "eta_seconds": 90.0,
  "message": "Analyzing window 12/84",
  "log": ["...", "..."]
}
```

### GET /api/analyses/{id}/video
Streams the uploaded video file.

### GET /api/analyses/{id}/export/json
Download JSON export.

### GET /api/analyses/{id}/export/csv
Download CSV export.

### POST /api/analyses/{id}/export/video
Body: `{ "line_thickness"?: number, "max_gap_frames"?: number, "show_markers"?: boolean }`
Returns: `{ "job_id": "...", "status": "processing" }` or streams/downloads MP4.
Prefer: returns downloadable MP4 directly when ready (sync for local) OR
`{ "download_url": "/api/analyses/{id}/export/video/download" }` after completion.

### GET /api/analyses/{id}/export/video/file
Streams burned-in MP4.

## Data Models

### Analysis
```json
{
  "id": "uuid",
  "name": "Session 1",
  "video_path": "relative/path",
  "video_url": "/api/analyses/{id}/video",
  "video_width": 1920,
  "video_height": 1080,
  "fps": 30.0,
  "duration": 12.5,
  "frame_count": 375,
  "created_at": "ISO-8601",
  "status": "created|uploaded|analyzing|completed|failed|cancelled",
  "model": "qwen3-vl:8b",
  "error": null,
  "detections": []
}
```

### Detection
```json
{
  "id": "uuid",
  "frame_index": 42,
  "timestamp": 1.4,
  "ball_visible": true,
  "x": 640.0,
  "y": 360.0,
  "normalized_x": 0.333,
  "normalized_y": 0.333,
  "confidence": 0.91,
  "source_window": 5
}
```

### AnalysisSummary
Same as Analysis but without `detections` array; includes `detection_count`.

## Errors

Every error response, whatever the status, has this shape:

```json
{ "detail": "That video is larger than the 500 MB limit. Trim or compress it, then upload it again.", "code": "upload_too_large" }
```

`detail` is plain language for the person who clicked the button: what happened and what to do next. It never contains exception text, status codes or validation-library output (those go to the backend log). `code` is for programs.

| `code` | Status | When |
|---|---|---|
| `upload_too_large` | 413 | Upload over `max_upload_mb` |
| `video_unreadable` | 400 | The upload is not a video OpenCV can read |
| `no_video` | 400 | Start or export on an analysis with no video |
| `analysis_not_found` | 404 | Unknown analysis id |
| `video_missing` | 404 | No video, or the file was removed from disk |
| `export_missing` | 404 | Download before an export exists |
| `ffmpeg_unavailable` | 503 | Video export needs FFmpeg and it is missing or failed |
| `invalid_request` | 422 | Request body failed validation |
| `not_found` | 404 | Unknown route |
| `internal_error` | 500 | Anything unexpected |

Ollama problems (not running, model not installed, timeout, unreadable answer) do not fail a request: they stop the analysis. It ends with `status` `failed` and `error` set to the plain-language message, and the same text is in the progress stream's `message`. The Ollama fields of `GET /api/settings/ollama` (`available`, `error`) use the same wording.

## CORS
Allow `http://localhost:5173` and `http://127.0.0.1:5173`.
