# BallTrace Backend

Local-first cricket ball tracking API. A local Ollama vision model
(default `qwen3-vl:8b`) is the **only** ball detector — OpenCV is used
solely for mechanical frame extraction and for drawing the trajectory
overlay on exported video. No YOLO/Detectron/color-threshold detection is
used anywhere in this service.

## Requirements

- Python 3.11 or 3.12
- [Ollama](https://ollama.com) running locally with a vision-capable model
  pulled (default expects `qwen3-vl:8b`: `ollama pull qwen3-vl:8b`)
- `ffmpeg` on `PATH` (optional, but recommended for broadly-compatible MP4
  exports; falls back to OpenCV's own MP4 encoder if missing)

## Setup

From the repo root, using the existing project virtualenv:

```bash
source .venv/bin/activate
pip install -r balltrace/backend/requirements.txt
```

## Run the API

```bash
cd balltrace/backend
python -m app
```

The API is served at `http://127.0.0.1:8000` (this computer only; there is no login, see the "Security model" in [`../README.md`](../README.md)). Host, port and the upload size limit come from `BALLTRACE_HOST`, `BALLTRACE_PORT` and `BALLTRACE_MAX_UPLOAD_MB`. All routes are under
`/api` (see `balltrace/API_CONTRACT.md`). CORS is open for
`http://localhost:5173` and `http://127.0.0.1:5173` (the frontend dev
server).

## Data storage

- SQLite database: `balltrace/data/balltrace.db` (analyses, detections,
  settings tables)
- Uploaded videos and exports: `balltrace/data/analyses/{analysis_id}/`

All paths are created automatically on startup.

## Run tests

```bash
cd balltrace/backend
pytest tests/ -v
```

Tests cover JSON parsing/repair of model output, overlapping-window merge
and dedupe, trajectory gap handling, normalized coordinate math,
frame/timestamp conversions, the Ollama client and the analysis pipeline
against a fake Ollama (`httpx.MockTransport`), and the HTTP error and upload
limit behaviour. They do not require Ollama, ffmpeg, or a real video file
to be present.

## Notes on the detection pipeline

1. The video is split into overlapping sliding windows (default size 8,
   overlap 4).
2. Each window's frames are JPEG-encoded and sent as a single `/api/chat`
   request to Ollama with the centralized prompt in
   `app/prompts/ball_detection.py`.
3. The model's JSON response is parsed with a tolerant extractor that
   strips markdown fences and can recover from truncated output; if
   parsing/validation still fails, one retry is made with a correction
   prompt. If that also fails, the window is skipped and the pipeline
   continues.
4. Detections from overlapping windows are merged per frame (preferring
   visible detections, then higher confidence), normalized coordinates and
   timestamps are computed, and results are persisted to SQLite.
5. Progress (including a cancel flag checked between windows) is exposed
   via `GET /api/analyses/{id}/progress` as Server-Sent Events.
