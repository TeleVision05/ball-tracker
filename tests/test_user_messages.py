"""Failures shown to a person must be plain language; the raw detail goes to the log."""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest

from balltracker.errors import (
    CalibrationFailedError,
    CalibrationInputError,
    CalibrationMissingError,
    NoBallFoundError,
    VideoOpenError,
)
from balltracker.flight_window import motion_energy_series
from balltracker.landmarks import LANDMARK_ORDER, calibrate_from_points
from balltracker.pipeline import analyze_from_track
from balltracker.scenario import protocol_camera
from balltracker.calibration import project_stumps_from_synthetic_camera
from balltracker.tracking import TrackPoint
from balltracker.user_messages import GENERIC_MESSAGE, describe_error, friendly_message

# Anything that looks like a developer artefact: a path, a number with units, a
# class name, a traceback, an HTTP status, a library name.
_TECHNICAL = re.compile(
    r"Traceback|Error\b|Exception|solvePnP|OpenCV|cv2|RMSE|\bpx\b|\bHTTP\b|\b[45]\d\d\b|/|\\",
)

KNOWN = [
    (NoBallFoundError("Insufficient track points (3) in /tmp/clip.mp4"), "ball"),
    (CalibrationInputError("Calibration RMSE 232.88 px exceeds 15 px limit."), "stump"),
    (CalibrationFailedError("solvePnP failed to estimate camera pose"), "camera"),
    (CalibrationMissingError("No camera calibration found for /tmp/clip.mp4"), "camera setup"),
    (VideoOpenError("Cannot open video: /tmp/clip.mp4"), "video"),
    (FileNotFoundError("/tmp/secret/session.json"), "file"),
]


@pytest.mark.parametrize(("exc", "keyword"), KNOWN, ids=lambda v: type(v).__name__)
def test_known_failures_get_plain_language(exc: BaseException, keyword: str) -> None:
    message = friendly_message(exc)
    assert keyword in message
    assert message != GENERIC_MESSAGE
    assert not _TECHNICAL.search(message), message
    assert str(exc) not in message


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("pydantic validation error for X: field required [type=missing]"),
        RuntimeError("HTTP 500 at /api/secret"),
        KeyError("internal_field"),
    ],
)
def test_unknown_failures_fall_back_to_generic(exc: BaseException) -> None:
    assert friendly_message(exc) == GENERIC_MESSAGE
    assert not _TECHNICAL.search(GENERIC_MESSAGE)


def test_describe_error_logs_raw_detail_and_returns_friendly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    exc = VideoOpenError("Cannot open video: /private/clip.mp4")
    with caplog.at_level(logging.ERROR, logger="balltracker.app"):
        message = describe_error(exc, "loading the preview frame")
    assert "/private/clip.mp4" not in message
    assert "/private/clip.mp4" in caplog.text
    assert "loading the preview frame" in caplog.text
    assert "VideoOpenError" in caplog.text


def test_unreadable_video_raises_the_typed_error(tmp_path) -> None:
    with pytest.raises(VideoOpenError):
        motion_energy_series(tmp_path / "missing.mp4")


def test_too_few_stump_clicks_raise_the_typed_error() -> None:
    points = {name: np.array([100.0 + i, 200.0]) for i, name in enumerate(LANDMARK_ORDER[:2])}
    with pytest.raises(CalibrationInputError):
        calibrate_from_points(points, image_size=(1280, 720), fov_deg=55.0)


@pytest.mark.parametrize("layout", ["one_spot", "straight_line"])
def test_degenerate_stump_clicks_raise_a_typed_error_not_an_opencv_one(layout: str) -> None:
    if layout == "one_spot":
        points = {name: np.array([100.0, 100.0]) for name in LANDMARK_ORDER[:6]}
    else:
        points = {
            name: np.array([100.0 + 50 * i, 300.0]) for i, name in enumerate(LANDMARK_ORDER[:6])
        }
    with pytest.raises(CalibrationFailedError):
        calibrate_from_points(points, image_size=(1280, 720), fov_deg=55.0)


def test_scattered_stump_clicks_are_rejected_as_input_error() -> None:
    points = {
        name: np.array([(i * 377) % 1200 + 0.0, (i * 911) % 700 + 0.0])
        for i, name in enumerate(LANDMARK_ORDER[:6])
    }
    with pytest.raises(CalibrationInputError):
        calibrate_from_points(points, image_size=(1280, 720), fov_deg=55.0)


def test_too_few_marked_ball_positions_raise_no_ball_found(tmp_path) -> None:
    calib = project_stumps_from_synthetic_camera(protocol_camera())
    track = [
        TrackPoint(
            frame_idx=i,
            t_s=i / 30,
            u=600.0 + i,
            v=300.0,
            radius_px=4.0,
            confidence=1.0,
            predicted=False,
        )
        for i in range(3)
    ]
    with pytest.raises(NoBallFoundError):
        analyze_from_track(tmp_path / "missing.mp4", track, calib=calib)
