"""Central configuration and default settings for the BallTrace backend.

All paths are resolved relative to this file so the app works regardless of
the current working directory it is launched from.
"""
from __future__ import annotations

from pathlib import Path

# balltrace/backend/app/config.py -> app -> backend -> balltrace
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
BALLTRACE_DIR = BACKEND_DIR.parent

DATA_DIR = BALLTRACE_DIR / "data"
ANALYSES_DIR = DATA_DIR / "analyses"
DB_PATH = DATA_DIR / "balltrace.db"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-vl:8b"
DEFAULT_WINDOW_SIZE = 4
DEFAULT_WINDOW_OVERLAP = 0
# Analyze every Nth frame (1 = every frame). Biggest speed lever for VL models.
DEFAULT_FRAME_STRIDE = 3

# Request timeout (seconds) for a single Ollama /api/chat call. Vision models
# on modest local hardware can be slow (especially multi-image windows), so
# this is generous.
OLLAMA_CHAT_TIMEOUT = 600.0
OLLAMA_HEALTH_TIMEOUT = 5.0

# Enough tokens for thinking VLMs (qwen3-vl) to finish residual reasoning and
# still emit JSON. Lower = faster; too low truncates mid-JSON.
OLLAMA_NUM_PREDICT = 768
OLLAMA_TEMPERATURE = 0.0

# Smaller JPEG payloads = faster prompt eval on the VLM.
JPEG_QUALITY = 55

# Frames are downscaled before Ollama (coords are scaled back afterwards).
OLLAMA_MAX_FRAME_WIDTH = 480

# Per-frame requests are far more reliable with thinking VLMs than stuffing
# many raw images into one chat turn. Unique frames are sampled by stride
# (no overlap re-work).
ANALYZE_FRAMES_INDIVIDUALLY = True

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

DEFAULT_EXPORT_LINE_THICKNESS = 3
DEFAULT_EXPORT_MAX_GAP_FRAMES = 10
DEFAULT_EXPORT_SHOW_MARKERS = True

# Neon cyan-ish trajectory color, approx #00FFD0, stored as BGR for OpenCV.
TRAJECTORY_COLOR_BGR = (208, 255, 0)

PROGRESS_POLL_INTERVAL_SECONDS = 1.0


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)


def analysis_dir(analysis_id: str) -> Path:
    return ANALYSES_DIR / analysis_id


def analysis_export_dir(analysis_id: str) -> Path:
    return analysis_dir(analysis_id) / "export"
