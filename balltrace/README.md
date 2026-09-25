# BallTrace

Local-first AI cricket ball tracking. Upload batting/net practice video, analyze frames with a local Ollama vision model (`qwen3-vl:8b` by default), and visualize the ball trajectory as a neon overlay on the video.

Ball detection is performed exclusively by the Ollama vision model — not YOLO, Detectron, or color thresholding.

## Architecture

| Layer | Stack |
|-------|--------|
| Frontend | React, TypeScript, Vite, Tailwind |
| Backend | Python, FastAPI |
| AI | Ollama HTTP API (vision) |
| Video | OpenCV frame extraction; FFmpeg for burned-in export |
| Storage | SQLite + local video files |

## Prerequisites

1. **Python 3.10+** and Node.js 18+
2. **[Ollama](https://ollama.com)** running locally with a vision model:

   ```bash
   ollama pull qwen3-vl:8b
   ```

3. **FFmpeg** (optional, required for burned-in video export):

   ```bash
   brew install ffmpeg   # macOS
   ```

## Quick start

From the BallTracker repo root:

```bash
# Backend
source .venv/bin/activate          # or: python3 -m venv .venv && source .venv/bin/activate
pip install -r balltrace/backend/requirements.txt
cd balltrace/backend
uvicorn app.main:app --reload --port 8000
```

In another terminal:

```bash
# Frontend
cd balltrace/frontend
npm install
npm run dev
```

Open **http://localhost:5173**

## Workflow

1. Click **New Analysis** and name the session.
2. Upload a cricket batting/net video.
3. Click **Analyze Ball** — frames are sent in sliding windows (default 8 frames, overlap 4) to Ollama.
4. Watch live progress; cancel anytime.
5. Scrub the video — the trajectory draws up to the current time.
6. Export detections as JSON/CSV or a burned-in MP4.

## Settings

Configure in the UI **Settings** panel:

- Ollama base URL (default `http://localhost:11434`)
- Model name (default `qwen3-vl:8b`)
- Sliding window size and overlap

## Tests

```bash
cd balltrace/backend
pytest tests/ -v
```

## Prompt

The vision prompt lives in one place for easy editing:

`backend/app/prompts/ball_detection.py`

## Data

Local only — no cloud:

- SQLite: `balltrace/data/balltrace.db`
- Videos & exports: `balltrace/data/analyses/{id}/`
