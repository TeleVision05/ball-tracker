"""SQLite persistence layer (aiosqlite) for analyses, detections, settings.

Kept intentionally simple: a fresh connection is opened per operation. For a
local, single-user tool this avoids any pooling/locking complexity while
still being safe (SQLite handles short-lived connections fine).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import aiosqlite

from app.config import (
    DATA_DIR,
    DB_PATH,
    DEFAULT_FRAME_STRIDE,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_WINDOW_OVERLAP,
    DEFAULT_WINDOW_SIZE,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    video_path TEXT,
    video_width INTEGER,
    video_height INTEGER,
    fps REAL,
    duration REAL,
    frame_count INTEGER,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    model TEXT,
    window_size INTEGER,
    window_overlap INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    ball_visible INTEGER NOT NULL,
    x REAL,
    y REAL,
    normalized_x REAL,
    normalized_y REAL,
    confidence REAL NOT NULL,
    source_window INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_detections_analysis_id ON detections(analysis_id);
CREATE INDEX IF NOT EXISTS idx_detections_frame_index ON detections(analysis_id, frame_index);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_DEFAULT_SETTINGS = {
    "base_url": DEFAULT_OLLAMA_URL,
    "model": DEFAULT_MODEL,
    "window_size": DEFAULT_WINDOW_SIZE,
    "window_overlap": DEFAULT_WINDOW_OVERLAP,
    "frame_stride": DEFAULT_FRAME_STRIDE,
}


async def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.executescript(_SCHEMA)
        await conn.commit()

        # Seed defaults for any settings keys that don't exist yet.
        for key, value in _DEFAULT_SETTINGS.items():
            await conn.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
        await conn.commit()


@asynccontextmanager
async def _connection():
    """Async context manager yielding a connected aiosqlite Connection.

    NOTE: aiosqlite's Connection object is itself awaitable *and* an async
    context manager, and both paths trigger the same one-shot "start the
    worker thread" logic. Combining them (`async with await conn:`) starts
    the thread twice and raises `RuntimeError: threads can only be started
    once`. This helper awaits exactly once to connect, then closes
    explicitly in a `finally`, so callers never trigger that twice.
    """
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    try:
        yield conn
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


async def create_analysis(record: Dict[str, Any]) -> None:
    async with _connection() as conn:
        await conn.execute(
            """
            INSERT INTO analyses (
                id, name, video_path, video_width, video_height, fps,
                duration, frame_count, created_at, status, model,
                window_size, window_overlap, error
            ) VALUES (:id, :name, :video_path, :video_width, :video_height, :fps,
                :duration, :frame_count, :created_at, :status, :model,
                :window_size, :window_overlap, :error)
            """,
            record,
        )
        await conn.commit()


async def update_analysis(analysis_id: str, **fields: Any) -> None:
    if not fields:
        return
    columns = ", ".join(f"{key} = :{key}" for key in fields)
    params = {**fields, "id": analysis_id}
    async with _connection() as conn:
        await conn.execute(f"UPDATE analyses SET {columns} WHERE id = :id", params)
        await conn.commit()


async def get_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    async with _connection() as conn:
        cursor = await conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def list_analyses() -> List[Dict[str, Any]]:
    async with _connection() as conn:
        cursor = await conn.execute("SELECT * FROM analyses ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def delete_analysis(analysis_id: str) -> None:
    async with _connection() as conn:
        await conn.execute("DELETE FROM detections WHERE analysis_id = ?", (analysis_id,))
        await conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        await conn.commit()


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


async def clear_detections(analysis_id: str) -> None:
    async with _connection() as conn:
        await conn.execute("DELETE FROM detections WHERE analysis_id = ?", (analysis_id,))
        await conn.commit()


async def insert_detections(analysis_id: str, detections: List[Dict[str, Any]]) -> None:
    if not detections:
        return
    rows = [
        {
            "id": d["id"],
            "analysis_id": analysis_id,
            "frame_index": d["frame_index"],
            "timestamp": d["timestamp"],
            "ball_visible": 1 if d["ball_visible"] else 0,
            "x": d.get("x"),
            "y": d.get("y"),
            "normalized_x": d.get("normalized_x"),
            "normalized_y": d.get("normalized_y"),
            "confidence": d.get("confidence", 0.0),
            "source_window": d.get("source_window", 0),
        }
        for d in detections
    ]
    async with _connection() as conn:
        await conn.executemany(
            """
            INSERT INTO detections (
                id, analysis_id, frame_index, timestamp, ball_visible,
                x, y, normalized_x, normalized_y, confidence, source_window
            ) VALUES (:id, :analysis_id, :frame_index, :timestamp, :ball_visible,
                :x, :y, :normalized_x, :normalized_y, :confidence, :source_window)
            """,
            rows,
        )
        await conn.commit()


async def get_detections(analysis_id: str) -> List[Dict[str, Any]]:
    async with _connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM detections WHERE analysis_id = ? ORDER BY frame_index ASC",
            (analysis_id,),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["ball_visible"] = bool(d["ball_visible"])
            results.append(d)
        return results


async def count_detections(analysis_id: str) -> int:
    async with _connection() as conn:
        cursor = await conn.execute(
            "SELECT COUNT(*) as c FROM detections WHERE analysis_id = ?", (analysis_id,)
        )
        row = await cursor.fetchone()
        return int(row["c"]) if row else 0


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


async def get_settings() -> Dict[str, Any]:
    async with _connection() as conn:
        cursor = await conn.execute("SELECT key, value FROM settings")
        rows = await cursor.fetchall()
    settings = dict(_DEFAULT_SETTINGS)
    for row in rows:
        settings[row["key"]] = json.loads(row["value"])
    return settings


async def update_settings(partial: Dict[str, Any]) -> Dict[str, Any]:
    async with _connection() as conn:
        for key, value in partial.items():
            if value is None:
                continue
            await conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )
        await conn.commit()
    return await get_settings()
