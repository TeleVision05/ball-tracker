# BallTrace

> **A separate experiment.** BallTrace lives in this repository but is not part of the `balltracker` analytics package: it has its own backend, frontend, dependencies and tests, and it detects the ball with an Ollama vision model rather than the stump-calibrated pipeline in `src/balltracker`.

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

1. **Python 3.11 or 3.12** and Node.js 22.22.2+ or 24.15+ (or 26+; the frontend test runner, jsdom, needs these)
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
python -m app                      # http://127.0.0.1:8000, this computer only
```

In another terminal:

```bash
# Frontend
cd balltrace/frontend
npm install
npm run dev
```

Open **http://localhost:5173**

## Security model

BallTrace is a local tool for one person on one computer. **There is no login.**

- The backend listens on `127.0.0.1` only. Anyone who can reach its port can upload videos, read every analysis and make your Ollama run work for them.
- **Do not expose it to a network.** Do not set `BALLTRACE_HOST` to `0.0.0.0` or a LAN address, do not put it behind a public tunnel or reverse proxy, and do not run it on a shared machine you do not trust. If you set `BALLTRACE_HOST` to a non-loopback address, the backend logs a warning at start-up.
- Uploads are limited to `BALLTRACE_MAX_UPLOAD_MB` (default 500). That limit keeps a mistaken drop from filling the disk; it is not a defence against someone hostile.
- Videos, the database and exports stay on disk under `balltrace/data/` (gitignored). Nothing is sent anywhere except frames to the Ollama URL you configure, which is `http://localhost:11434` by default.
- Errors shown in the browser are plain language; the technical detail is in the backend's log output.

## Workflow

1. Click **New Analysis** and name the session.
2. Upload a cricket batting/net video.
3. Click **Analyze Ball** — every 3rd frame (`DEFAULT_FRAME_STRIDE` in `backend/app/config.py`) is sent to Ollama one at a time; the sliding-window mode (default window 4, overlap 0) is used only when `ANALYZE_FRAMES_INDIVIDUALLY` is off.
4. Watch live progress; cancel anytime.
5. Scrub the video — the trajectory draws up to the current time.
6. Export detections as JSON/CSV or a burned-in MP4.

## Configuration

Environment variables, read when the backend starts:

| Variable | Default | What it does |
|---|---|---|
| `BALLTRACE_HOST` | `127.0.0.1` | Address `python -m app` listens on. See the security model above before changing it. |
| `BALLTRACE_PORT` | `8000` | Port `python -m app` listens on. The frontend dev server proxies `/api` to `127.0.0.1:8000`. |
| `BALLTRACE_MAX_UPLOAD_MB` | `500` | Largest video accepted. Bigger files are refused with a message that names the limit. The frontend reads it from `GET /api/limits`. |

A value that is not a positive whole number is ignored and the default is used.

## Settings

Configure in the UI **Settings** panel:

- Ollama base URL (default `http://localhost:11434`)
- Model name (default `qwen3-vl:8b`)
- Sliding window size and overlap (only used when `ANALYZE_FRAMES_INDIVIDUALLY` is off)

## Tests

```bash
cd balltrace/backend && pytest -q          # needs no Ollama, ffmpeg or video file
cd balltrace/frontend && npm test          # vitest + React Testing Library
```

The Ollama path is tested against a fake server (`httpx.MockTransport`): success, model not installed, server down, malformed replies and timeouts, for both `OllamaClient` and the `run_analysis` pipeline. Nothing in the tests talks to a real Ollama, so behaviour with a real model is checked by hand.

## When something goes wrong

Messages are worded in one place per layer (`backend/app/friendly_errors.py`, `frontend/src/lib/friendlyError.ts`). Every error response is `{"detail": "<plain-language message>", "code": "<machine code>"}`; the raw exception is logged by the backend. If an analysis stops, the message says what to fix (start Ollama, install the model, shrink the video).

## Prompt

The vision prompt lives in one place for easy editing:

`backend/app/prompts/ball_detection.py`

## Data

Local only — no cloud:

- SQLite: `balltrace/data/balltrace.db`
- Videos & exports: `balltrace/data/analyses/{id}/`
