"""Exact regression pins for the hybrid tracker on in-memory synthetic clips.

The clip-based characterisation tests use tolerances so they survive video
codec differences. These run the tracker on frames rendered in memory
(``FrameCapture``, no codec) and compare every track point with values
recorded from the tracker before it was split into modules, so a change to a
gate size, a threshold or a confidence shows up here.

If you change the tracker on purpose, regenerate the pins and say why in the
commit: ``python tests/test_tracker_pins.py --write``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import pytest

from balltracker.config import load_config
from balltracker.release import ReleaseSeed
from balltracker.tracking import track_delivery_hybrid, track_video
from synthetic_clips import FakeCSRT, FrameCapture

PINS = Path(__file__).parent / "data" / "hybrid_tracker_pins.json"


def _capture(path: str) -> FrameCapture:
    """``mem://<seed>[/hard]`` -> an in-memory clip."""
    seed, _, kind = path.removeprefix("mem://").partition("/")
    return FrameCapture(int(seed), hard=kind == "hard")


def _config(**overrides) -> dict:
    config = load_config()
    config["detection"]["use_pose_release"] = False
    config["detection"].update(overrides)
    return config


def _seed(path: str, frame_idx: int = 3) -> ReleaseSeed:
    u, v = _capture(path).uv[frame_idx]
    return ReleaseSeed(frame_idx, float(u), float(v), (0.0, 1.0), 0.8)


def _scenarios() -> dict[str, object]:
    return {
        "plain42_nopose": lambda: track_delivery_hybrid("mem://42", fps=120, config=_config()),
        "plain2026_seed": lambda: track_delivery_hybrid(
            "mem://2026", fps=120, config=_config(), release_seed=_seed("mem://2026")
        ),
        "plain42_seed5": lambda: track_delivery_hybrid(
            "mem://42", fps=120, config=_config(), release_seed=_seed("mem://42", 5)
        ),
        "hard42_gate25_seed": lambda: track_delivery_hybrid(
            "mem://42/hard",
            fps=120,
            config=_config(release_max_gate_px=25.0),
            release_seed=_seed("mem://42/hard"),
        ),
        "hard2026_gate20_seed": lambda: track_delivery_hybrid(
            "mem://2026/hard",
            fps=120,
            config=_config(release_max_gate_px=20.0, use_csrt=False),
            release_seed=_seed("mem://2026/hard"),
        ),
        "plain7_window": lambda: track_delivery_hybrid(
            "mem://7", config=_config(), start_frame=10, end_frame=40
        ),
        "hard42_seed_csrt": lambda: track_delivery_hybrid(
            "mem://42/hard", fps=120, config=_config(), release_seed=_seed("mem://42/hard")
        ),
        "hard2026_tight_gate": lambda: track_delivery_hybrid(
            "mem://2026/hard", fps=120, config=_config(release_max_gate_px=60.0)
        ),
        "plain7_hsv": lambda: track_video("mem://7", fps=120, config=_config(), method="hsv"),
    }


def _row(tp) -> list:
    return [
        tp.frame_idx,
        round(tp.t_s, 6),
        round(tp.u, 4),
        round(tp.v, 4),
        round(tp.radius_px, 4),
        round(tp.confidence, 4),
        tp.predicted,
    ]


def _run_all() -> dict[str, list[list]]:
    FakeCSRT.max_updates = 3
    return {name: [_row(tp) for tp in run()] for name, run in _scenarios().items()}


@pytest.fixture
def in_memory_video(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", _capture)
    monkeypatch.setattr(cv2, "TrackerCSRT_create", FakeCSRT, raising=False)
    monkeypatch.setattr(FakeCSRT, "max_updates", 3)


@pytest.mark.parametrize("name", list(_scenarios()))
def test_hybrid_tracker_matches_pins(in_memory_video, name: str) -> None:
    expected = json.loads(PINS.read_text(encoding="utf-8"))[name]
    actual = [_row(tp) for tp in _scenarios()[name]()]
    assert [r[0] for r in actual] == [r[0] for r in expected], "track frames changed"
    assert [r[6] for r in actual] == [r[6] for r in expected], "predicted flags changed"
    for got, want in zip(actual, expected):
        assert got[1:6] == pytest.approx(want[1:6], abs=1e-3), f"frame {got[0]}"


if __name__ == "__main__" and "--write" in sys.argv:
    cv2.VideoCapture = _capture
    cv2.TrackerCSRT_create = FakeCSRT
    PINS.parent.mkdir(exist_ok=True)
    PINS.write_text(json.dumps(_run_all(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {PINS}")
