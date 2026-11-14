"""Shared fixtures: an isolated database, a tiny video, and a fake Ollama server."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import cv2
import httpx
import numpy as np
import pytest

from app import config
from app.models import database as db
from app.services import progress

VIDEO_WIDTH = 160
VIDEO_HEIGHT = 120
VIDEO_FRAMES = 10


def run(coro):
    """Run a coroutine to completion (the backend tests avoid an asyncio plugin)."""
    return asyncio.run(coro)


@pytest.fixture
def isolated_data(tmp_path, monkeypatch) -> Path:
    """Point the SQLite file and analysis folders at a temp directory."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "ANALYSES_DIR", tmp_path / "analyses")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "balltrace.db")
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "balltrace.db")
    run(db.init_db())
    return tmp_path


@pytest.fixture
def tiny_video(tmp_path) -> Path:
    """A real, readable 10-frame MP4 (frames differ so decoding is observable)."""
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (VIDEO_WIDTH, VIDEO_HEIGHT)
    )
    assert writer.isOpened()
    for i in range(VIDEO_FRAMES):
        frame = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def make_analysis(analysis_id: str, video: Path) -> Dict[str, Any]:
    return {
        "id": analysis_id,
        "name": "test",
        "video_path": str(video),
        "video_width": VIDEO_WIDTH,
        "video_height": VIDEO_HEIGHT,
        "fps": 10.0,
        "duration": 1.0,
        "frame_count": VIDEO_FRAMES,
        "created_at": "2026-01-01T00:00:00+00:00",
        "status": "analyzing",
        "model": "fake-model",
        "window_size": 4,
        "window_overlap": 0,
        "error": None,
    }


@pytest.fixture
def analysis_with_video(isolated_data, tiny_video) -> str:
    analysis_id = "analysis-1"
    run(db.create_analysis(make_analysis(analysis_id, tiny_video)))
    yield analysis_id
    progress.clear(analysis_id)


def chat_reply(detections: List[Dict[str, Any]], *, wrap_in_fence: bool = False) -> httpx.Response:
    """A successful /api/chat reply carrying a detections JSON payload."""
    text = json.dumps({"detections": detections})
    if wrap_in_fence:
        text = f"```json\n{text}\n```"
    return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})


def ball_at(x: float, y: float, frame_index: int = 0) -> Dict[str, Any]:
    return {"frame_index": frame_index, "ball_visible": True, "x": x, "y": y, "confidence": 0.9}


def text_reply(text: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})


def model_missing_reply(model: str = "fake-model") -> httpx.Response:
    return httpx.Response(404, json={"error": f'model "{model}" not found, try pulling it first'})


class FakeOllama:
    """An ``httpx.MockTransport`` that records requests and answers from a script.

    ``script`` is a list of replies (``httpx.Response`` or an exception to
    raise); the last entry repeats once the list is exhausted.
    """

    def __init__(self, script: List[Any]):
        self.script = list(script)
        self.requests: List[httpx.Request] = []
        self.on_request: Callable[[int], None] | None = None
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.on_request is not None:
            self.on_request(len(self.requests))
        step = self.script[min(len(self.requests) - 1, len(self.script) - 1)]
        if isinstance(step, Exception):
            raise step
        return step

    def chat_payloads(self) -> List[Dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == "/api/chat"]


def decode_images(payload: Dict[str, Any]) -> List[bytes]:
    return [base64.b64decode(img) for img in payload["messages"][0]["images"]]
