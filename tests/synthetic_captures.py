"""Synthetic Android captures (mp4 plus version-2 sidecar) shared by tests.

Imported as a top-level module (``from synthetic_captures import ...``), like
``synthetic_clips``: pytest puts ``tests/`` on ``sys.path``, while a
``tests.`` package import only works when the repo root happens to be on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from balltracker.pitch import stump_landmarks_3d
from balltracker.synthetic import default_camera, generate_delivery


def timestamp_span_s(timestamps: list[int]) -> float:
    if len(timestamps) < 2:
        return 0.0
    return (int(timestamps[-1]) - int(timestamps[0])) / 1_000_000


def sidecar_payload(
    *,
    delivery_id: str,
    video_file: str,
    fps: int,
    frame_count: int,
    timestamps: list[int] | None = None,
) -> dict:
    camera = default_camera()
    width, height = camera.resolution
    landmarks = stump_landmarks_3d()
    marker_names = {
        "nearLeft": "bowler_left_base",
        "nearRight": "bowler_right_base",
        "farLeft": "batter_left_base",
        "farRight": "batter_right_base",
    }
    markers = {}
    for marker, landmark in marker_names.items():
        pixel = camera.project(landmarks[landmark].reshape(1, 3))[0]
        markers[marker] = [float(pixel[0] / width), float(pixel[1] / height)]
    if timestamps is None:
        timestamps = [round(i * 1_000_000 / fps) for i in range(frame_count)]
    span_s = timestamp_span_s(timestamps)
    return {
        "version": 2,
        "id": delivery_id,
        "videoFile": video_file,
        "encodedWidth": width,
        "encodedHeight": height,
        "uprightWidth": width,
        "uprightHeight": height,
        "orientationHintDegrees": 0,
        "measuredFps": float(fps),
        "captureFps": fps,
        "wallClockSeconds": span_s,
        "durationSeconds": span_s,
        "presentationTimestampsUs": timestamps,
        "markers": markers,
        "camera": {"intrinsicMatrixUpright": camera.intrinsic_matrix().tolist()},
    }


def write_synthetic_capture(out_dir: Path, fps: int = 120, seed: int = 7) -> tuple[Path, dict]:
    """Render one delivery and write a version-2 sidecar beside the mp4.

    Seed 7 keeps the red ball in frame from release through the bounce, so HSV
    tracking covers the flight the speed and bounce checks measure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    delivery_id = "delivery_0001"
    gt = generate_delivery(delivery_id, out_dir, fps, np.random.default_rng(seed))
    payload = sidecar_payload(
        delivery_id=gt["delivery_id"],
        video_file=f"{delivery_id}.mp4",
        fps=fps,
        frame_count=int(gt["frame_count"]),
    )
    sidecar_path = out_dir / f"{delivery_id}.capture.json"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar_path, gt
