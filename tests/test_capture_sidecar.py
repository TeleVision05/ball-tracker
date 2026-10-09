"""Tests for Android capture sidecars and planar camera calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from balltracker.calibration import reprojection_rmse
from balltracker.capture_sidecar import (
    apply_presentation_times,
    calibrate_from_capture,
    capture_sidecar_from_dict,
    load_capture_sidecar,
    marker_pixels,
    presentation_times_s,
)
from balltracker.pitch import stump_landmarks_3d
from balltracker.synthetic import default_camera
from balltracker.tracking import TrackPoint


def test_rejects_version_1() -> None:
    with pytest.raises(ValueError, match=r"version 1"):
        capture_sidecar_from_dict({"version": 1})


def test_missing_intrinsic_matrix() -> None:
    with pytest.raises(ValueError, match="upright intrinsic matrix"):
        capture_sidecar_from_dict({"version": 2, "camera": {}})
    with pytest.raises(ValueError, match="upright intrinsic matrix"):
        capture_sidecar_from_dict(
            {
                "version": 2,
                "camera": {"intrinsicMatrixUpright": [[1.0, 0.0], [0.0, 1.0]]},
            }
        )


def test_round_trip_recovers_default_camera(tmp_path: Path) -> None:
    camera = default_camera()
    landmarks = stump_landmarks_3d()
    marker_names = {
        "nearLeft": "bowler_left_base",
        "nearRight": "bowler_right_base",
        "farLeft": "batter_left_base",
        "farRight": "batter_right_base",
    }
    width, height = camera.resolution
    markers = {}
    for marker, landmark in marker_names.items():
        pixel = camera.project(landmarks[landmark].reshape(1, 3))[0]
        markers[marker] = [float(pixel[0] / width), float(pixel[1] / height)]

    payload = {
        "version": 2,
        "id": "round-trip",
        "videoFile": "clip.mp4",
        "encodedWidth": width,
        "encodedHeight": height,
        "uprightWidth": width,
        "uprightHeight": height,
        "orientationHintDegrees": 0,
        "measuredFps": 120.0,
        "captureFps": 120,
        "presentationTimestampsUs": [0, 8333, 16666],
        "markers": markers,
        "camera": {"intrinsicMatrixUpright": camera.intrinsic_matrix().tolist()},
    }
    sidecar = capture_sidecar_from_dict(payload)
    path = tmp_path / "clip.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_capture_sidecar(path)
    assert loaded.id == sidecar.id
    assert loaded.uprightWidth == width
    assert loaded.uprightHeight == height

    pixels = marker_pixels(sidecar)
    for marker, landmark in marker_names.items():
        expected = camera.project(landmarks[landmark].reshape(1, 3))[0]
        np.testing.assert_allclose(pixels[landmark], expected, atol=1e-6)

    calib = calibrate_from_capture(sidecar)
    center_error = np.linalg.norm(calib.camera_center_world - camera.position_m)
    assert center_error < 0.25
    assert calib.rms_reprojection_error < 2.0
    assert reprojection_rmse(calib, pixels, landmarks) < 2.0


def test_orientation_90_derives_upright_size() -> None:
    payload = _complete_sidecar(
        encodedWidth=1920,
        encodedHeight=1080,
        uprightWidth=None,
        uprightHeight=None,
        orientationHintDegrees=90,
    )
    sidecar = capture_sidecar_from_dict(payload)
    assert sidecar.uprightWidth == 1080
    assert sidecar.uprightHeight == 1920
    assert sidecar.orientationHintDegrees == 90


def test_presentation_times_s() -> None:
    assert presentation_times_s([]) == []
    assert presentation_times_s([0, 10000, 25000]) == [0.0, 0.01, 0.025]
    assert presentation_times_s([5, 5, 10005]) == [0.0, 0.0, 0.01]


def test_apply_presentation_times() -> None:
    track = [
        TrackPoint(0, 1.5, 10.0, 20.0, 4.0, 0.9, False),
        TrackPoint(2, 1.5, 12.0, 22.0, 4.0, 0.8, False),
        TrackPoint(9, 3.25, 14.0, 24.0, 4.0, 0.7, True),
    ]
    updated = apply_presentation_times(track, [0, 8000, 20000])
    assert updated[0].t_s == 0.0
    assert updated[1].t_s == 0.02
    assert updated[2].t_s == 3.25
    assert [point.t_s for point in track] == [1.5, 1.5, 3.25]
    assert updated[0] is not track[0]
    assert updated[2].predicted is True


def _complete_sidecar(**overrides: object) -> dict:
    payload = {
        "version": 2,
        "id": "orient",
        "videoFile": "clip.mp4",
        "encodedWidth": 1920,
        "encodedHeight": 1080,
        "uprightWidth": 1920,
        "uprightHeight": 1080,
        "orientationHintDegrees": 0,
        "measuredFps": None,
        "captureFps": 240,
        "presentationTimestampsUs": [],
        "markers": {
            "nearLeft": [0.2, 0.8],
            "nearRight": [0.8, 0.8],
            "farLeft": [0.4, 0.3],
            "farRight": [0.6, 0.3],
        },
        "camera": {
            "intrinsicMatrixUpright": [
                [1000.0, 0.0, 960.0],
                [0.0, 1000.0, 540.0],
                [0.0, 0.0, 1.0],
            ]
        },
    }
    payload.update(overrides)
    return payload
