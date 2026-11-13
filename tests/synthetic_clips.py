"""Seeded synthetic clips for the tracker characterisation tests.

``make_clip`` renders one delivery with :func:`balltracker.synthetic.generate_delivery`.
``make_hard_clip`` rewrites that clip so the tracker has to work for it: a
static textured background, a few frames where the ball is hidden (so the
Kalman filter and CSRT have to bridge a gap) and a white distractor that moves
across the top of the frame. Everything is derived from the seed, so the same
seed gives the same clip on the same OpenCV build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from balltracker.synthetic import generate_delivery

FPS = 120


@dataclass(frozen=True)
class Clip:
    """A rendered clip and its ground truth."""

    video: Path
    track_2d: np.ndarray  # (n_frames, 2) true ball centre per frame
    bounce_frame: int
    frame_count: int


def make_clip(out_dir: Path, seed: int) -> Clip:
    """Render one seeded synthetic delivery (red ball on black, stumps drawn)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = generate_delivery(f"clip_{seed}", out_dir, fps=FPS, rng=np.random.default_rng(seed))
    return Clip(
        video=out_dir / f"clip_{seed}.mp4",
        track_2d=np.asarray(ann["track_2d_px"], dtype=np.float64),
        bounce_frame=int(ann["bounce_frame_index"]),
        frame_count=int(ann["frame_count"]),
    )


def make_tiny_clip(out_dir: Path, n_frames: int = 3) -> Path:
    """A few black frames with a red dot: shorter than the tracker's warm-up."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240))
    for k in range(n_frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (100 + 10 * k, 60 + 12 * k), 6, (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    return path


class FakeCSRT:
    """Deterministic stand-in for OpenCV's CSRT tracker.

    opencv-python (without contrib) ships no CSRT, so the tracker's CSRT branch
    never runs on a stock install. Tests install this through
    ``cv2.TrackerCSRT_create`` to pin that branch. Each update drifts the box
    ``step`` pixels down; after ``max_updates`` updates it reports a loss.
    ``fail_init`` makes ``init`` raise, as a broken OpenCV build can.
    """

    max_updates = 2
    step = 6.0
    fail_init = False
    created = 0

    def __init__(self) -> None:
        type(self).created += 1
        self.bbox: tuple[float, float, float, float] | None = None
        self.updates = 0

    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        if self.fail_init:
            raise RuntimeError("fake CSRT init failure")
        self.bbox = tuple(float(x) for x in bbox)

    def update(self, frame: np.ndarray) -> tuple[bool, tuple[float, ...] | None]:
        self.updates += 1
        if self.bbox is None or self.updates > self.max_updates:
            return False, None
        x, y, w, h = self.bbox
        self.bbox = (x, y + self.step, w, h)
        return True, self.bbox


def make_hard_clip(out_dir: Path, seed: int, hidden: tuple[int, ...] = (20, 21, 22)) -> Clip:
    """Render ``make_clip`` again with texture, hidden frames and a distractor."""
    clip = make_clip(out_dir / "plain", seed)
    rng = np.random.default_rng(seed + 1)
    cap = cv2.VideoCapture(str(clip.video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    texture = rng.integers(0, 40, size=(h, w, 3), dtype=np.uint8)
    out_path = out_dir / f"hard_{seed}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in hidden:
            u, v = clip.track_2d[frame_idx]
            cv2.circle(frame, (int(u), int(v)), 14, (0, 0, 0), -1)
        frame = cv2.add(frame, texture)
        x = int(40 + 9 * frame_idx) % max(1, w - 60)
        cv2.rectangle(frame, (x, 30), (x + 18, 48), (255, 255, 255), -1)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    meta = {"seed": seed, "hidden": list(hidden), "frames": frame_idx}
    (out_dir / f"hard_{seed}.json").write_text(json.dumps(meta), encoding="utf-8")
    return Clip(
        video=out_path,
        track_2d=clip.track_2d,
        bounce_frame=clip.bounce_frame,
        frame_count=frame_idx,
    )
