"""Tests for analyzing an Android capture sidecar into delivery stats."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from balltracker.capture_analysis import analyze_capture, upright_video
from balltracker.pitch import stump_landmarks_3d
from balltracker.synthetic import default_camera, generate_delivery


def _sidecar_payload(
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
        "presentationTimestampsUs": timestamps,
        "markers": markers,
        "camera": {"intrinsicMatrixUpright": camera.intrinsic_matrix().tolist()},
    }


def _write_synthetic_capture(out_dir: Path, fps: int = 120, seed: int = 7) -> tuple[Path, dict]:
    """Render one delivery and write a version-2 sidecar beside the mp4.

    Seed 7 keeps the red ball in frame from release through the bounce, so HSV
    tracking covers the flight the speed and bounce checks measure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    delivery_id = "delivery_0001"
    gt = generate_delivery(delivery_id, out_dir, fps, np.random.default_rng(seed))
    payload = _sidecar_payload(
        delivery_id=gt["delivery_id"],
        video_file=f"{delivery_id}.mp4",
        fps=fps,
        frame_count=int(gt["frame_count"]),
    )
    sidecar_path = out_dir / f"{delivery_id}.capture.json"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar_path, gt


def test_missing_video_raises(tmp_path: Path) -> None:
    payload = _sidecar_payload(
        delivery_id="missing-clip",
        video_file="missing.mp4",
        fps=120,
        frame_count=3,
    )
    sidecar_path = tmp_path / "missing.capture.json"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="video file is required"):
        analyze_capture(sidecar_path)


def test_analyze_synthetic_capture(tmp_path: Path) -> None:
    sidecar_path, gt = _write_synthetic_capture(tmp_path / "clip")
    result = analyze_capture(sidecar_path, detection_method="hsv")

    gt_speed = float(gt["radar_speed_kmh"])
    est_speed = float(result["metrics"]["release_speed_kmh"])
    speed_error = abs(est_speed - gt_speed) / gt_speed
    gt_bounce = gt["bounce_position_m"]
    bounce_error = float(
        np.linalg.norm(
            [
                result["metrics"]["pitch_line_m"] - gt_bounce[0],
                result["metrics"]["bounce_y_m"] - gt_bounce[1],
            ]
        )
    )
    assert speed_error <= 0.15, f"release-speed error {speed_error:.3f}"
    assert bounce_error <= 1.5, f"horizontal bounce error {bounce_error:.3f} m"
    assert result["timing_source"] == "presentation_timestamps"
    assert "speed_text" in result["summary"]
    assert "bounce_text" in result["summary"]
    assert result["calib_rmse_px"] < 2


def test_empty_timestamps_use_frame_index(tmp_path: Path) -> None:
    out_dir = tmp_path / "clip"
    sidecar_path, _gt = _write_synthetic_capture(out_dir)
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["presentationTimestampsUs"] = []
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    result = analyze_capture(sidecar_path, detection_method="hsv")
    assert result["timing_source"] == "frame_index"
    assert "No presentation timestamps; times use frame index divided by fps." in result["warnings"]


def test_partial_timestamps_keep_frame_index_clock(tmp_path: Path) -> None:
    sidecar_path, _gt = _write_synthetic_capture(tmp_path / "clip")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    fps = int(payload["captureFps"])
    payload["presentationTimestampsUs"] = [0, round(1_000_000 / fps)]
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    result = analyze_capture(sidecar_path, detection_method="hsv")
    assert len(result["track"]) > 2
    assert result["timing_source"] == "frame_index"
    assert "Some track frames have no presentation timestamp." in result["warnings"]
    for point in result["track"]:
        assert point["t_s"] == pytest.approx(point["frame_idx"] / fps)


def test_irregular_timestamps_set_presentation_clock(tmp_path: Path) -> None:
    sidecar_path, gt = _write_synthetic_capture(tmp_path / "clip")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    fps = int(payload["captureFps"])
    timestamps = [
        0 if i == 0 else round(i * 1_000_000 / fps) + (5000 if i % 2 else 0)
        for i in range(int(gt["frame_count"]))
    ]
    payload["presentationTimestampsUs"] = timestamps
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    result = analyze_capture(sidecar_path, detection_method="hsv")
    assert result["timing_source"] == "presentation_timestamps"
    for point in result["track"]:
        expected = (timestamps[point["frame_idx"]] - timestamps[0]) / 1_000_000
        assert point["t_s"] == pytest.approx(expected, abs=1e-6)


def test_rotate_4x2_ninety_clockwise_becomes_2x4(tmp_path: Path) -> None:
    source = tmp_path / "wide.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (4, 2))
    assert writer.isOpened()
    writer.write(np.zeros((2, 4, 3), dtype=np.uint8))
    writer.release()

    upright, size, temporary = upright_video(source, 90)
    try:
        assert size == (2, 4)
        capture = cv2.VideoCapture(str(upright))
        ok, frame = capture.read()
        capture.release()
        assert ok
        assert frame.shape[1] == 2
        assert frame.shape[0] == 4
    finally:
        if temporary:
            upright.unlink(missing_ok=True)
