"""Session persistence for multi-delivery analysis workflows."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .config import load_config
from .viz import render_overlay_video, save_pitch_map_png

_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "sessions"


@dataclass
class Session:
    """One analysis session with config snapshot and delivery records."""

    id: str
    created_at: str
    config: dict
    root: Path
    deliveries: list[dict] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.root / self.id

    def save(self) -> None:
        """Persist session metadata to disk."""
        self.path.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": self.id,
            "created_at": self.created_at,
            "config": self.config,
            "deliveries": self.deliveries,
        }
        with (self.path / "session.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def create_session(root: str | Path | None = None) -> Session:
    """Create a new session folder under ``sessions/<id>/``."""
    if root is None:
        root = _DEFAULT_ROOT
    root = Path(root)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    created_at = datetime.now(timezone.utc).isoformat()
    config = load_config()
    session = Session(
        id=session_id,
        created_at=created_at,
        config=config,
        root=root,
        deliveries=[],
    )
    session.path.mkdir(parents=True, exist_ok=True)
    session.save()
    return session


def _next_delivery_id(session: Session) -> str:
    n = len(session.deliveries) + 1
    return f"delivery_{n:04d}"


def _delivery_dir(session: Session, delivery_id: str) -> Path:
    d = session.path / "deliveries" / delivery_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bounce_frame_from_analysis(analysis: dict) -> int | None:
    fit = analysis.get("fit", {})
    bounce_time = fit.get("bounce_time_s")
    track = analysis.get("track", [])
    if bounce_time is None or not track:
        return None
    # Nearest track frame to bounce time.
    best_idx = 0
    best_dt = float("inf")
    for tp in track:
        dt = abs(float(tp["t_s"]) - float(bounce_time))
        if dt < best_dt:
            best_dt = dt
            best_idx = int(tp["frame_idx"])
    return best_idx


def add_delivery_analysis(
    session: Session,
    video_path: str | Path,
    analysis_dict: dict[str, Any],
    write_clip: bool = True,
) -> str:
    """Store analysis JSON, optional overlay clip, and pitch-map thumbnail."""
    video_path = Path(video_path).resolve()
    delivery_id = _next_delivery_id(session)
    delivery_dir = _delivery_dir(session, delivery_id)

    analysis_path = delivery_dir / "analysis.json"
    with analysis_path.open("w", encoding="utf-8") as f:
        json.dump(analysis_dict, f, indent=2)

    overlay_path: Path | None = None
    pitch_map_path = delivery_dir / "pitch_map.png"

    metrics = analysis_dict.get("metrics", analysis_dict)
    save_pitch_map_png(metrics, pitch_map_path)

    if write_clip:
        track = analysis_dict.get("track", [])
        bounce_frame = _bounce_frame_from_analysis(analysis_dict)
        overlay_path = delivery_dir / "overlay.mp4"
        try:
            render_overlay_video(
                video_path,
                track,
                overlay_path,
                bounce_frame=bounce_frame,
                clip=True,
            )
        except Exception:
            # Fallback: copy source and force browser-compatible encoding.
            shutil.copy2(video_path, overlay_path)
            from .viz import ensure_web_playable

            ensure_web_playable(overlay_path)

    record = {
        "delivery_id": delivery_id,
        "source_video_path": str(video_path),
        "analysis_path": str(analysis_path),
        "overlay_path": str(overlay_path) if overlay_path else None,
        "pitch_map_path": str(pitch_map_path),
        "metrics": metrics,
        "checks": list(analysis_dict.get("checks", [])),
        "added_at": datetime.now(timezone.utc).isoformat(),
    }
    session.deliveries.append(record)
    session.save()
    return delivery_id


def list_deliveries(session: Session) -> list[dict]:
    """Return delivery metadata records for a session."""
    if session.deliveries:
        return list(session.deliveries)

    deliveries_dir = session.path / "deliveries"
    if not deliveries_dir.exists():
        return []

    records: list[dict] = []
    for sub in sorted(deliveries_dir.iterdir()):
        if not sub.is_dir():
            continue
        analysis_path = sub / "analysis.json"
        if not analysis_path.exists():
            continue
        with analysis_path.open(encoding="utf-8") as f:
            analysis = json.load(f)
        records.append(
            {
                "delivery_id": sub.name,
                "source_video_path": analysis.get("video_path"),
                "analysis_path": str(analysis_path),
                "overlay_path": str(sub / "overlay.mp4")
                if (sub / "overlay.mp4").exists()
                else None,
                "pitch_map_path": str(sub / "pitch_map.png")
                if (sub / "pitch_map.png").exists()
                else None,
                "metrics": analysis.get("metrics", {}),
            }
        )
    return records


def load_session(path: str | Path) -> Session:
    """Load session from a session directory or ``session.json`` path."""
    path = Path(path)
    if path.is_file() and path.name == "session.json":
        session_dir = path.parent
    else:
        session_dir = path

    session_json = session_dir / "session.json"
    if not session_json.exists():
        raise FileNotFoundError(f"No session.json in {session_dir}")

    with session_json.open(encoding="utf-8") as f:
        data = json.load(f)

    session = Session(
        id=data["id"],
        created_at=data["created_at"],
        config=data.get("config", {}),
        root=session_dir.parent,
        deliveries=data.get("deliveries", []),
    )
    if not session.deliveries:
        session.deliveries = list_deliveries(session)
    return session


def session_summary_df(session: Session) -> pd.DataFrame:
    """Tabular summary of per-delivery metrics."""
    rows: list[dict] = []
    for rec in list_deliveries(session):
        m = rec.get("metrics", {})
        rows.append(
            {
                "delivery_id": rec.get("delivery_id"),
                "release_speed_kmh": m.get("release_speed_kmh"),
                "beta_s_deg": m.get("beta_s_deg"),
                "turn_deg": m.get("turn_deg"),
                "pitch_line_m": m.get("pitch_line_m"),
                "pitch_length_m": m.get("pitch_length_m"),
                "source_video": rec.get("source_video_path"),
            }
        )
    return pd.DataFrame(rows)


def session_to_dict(session: Session) -> dict:
    """Serialize session dataclass (excluding non-JSON paths if needed)."""
    d = asdict(session)
    d["root"] = str(session.root)
    d["path"] = str(session.path)
    return d
