"""Regressions for defects found by running the pipeline on real clips.

docs/REAL_FOOTAGE.md records the runs. Every test here uses a small synthetic
stand-in; no real footage is read or committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import balltracker.benchmark as benchmark
from balltracker.app.result_step import delivery_checks
from balltracker.calibration import project_stumps_from_synthetic_camera
from balltracker.capture_analysis import analyze_capture, upright_video
from balltracker.config import load_config
from balltracker.metrics import compute_metrics, metrics_to_dict
from balltracker.physics import fit_trajectory
from balltracker.result_checks import AT_FIT_LIMIT, POOR_FIT, result_checks
from balltracker.scenario import protocol_camera
from balltracker.synthetic import default_camera
from balltracker.tracking import track_from_clicks
from synthetic_captures import write_synthetic_capture


def _textured_frames(n: int, width: int = 64, height: int = 48) -> list[np.ndarray]:
    """Grass-like noise with a 3 px red ball moving across it (upright frames)."""
    rng = np.random.default_rng(3)
    frames = []
    for k in range(n):
        frame = rng.integers(40, 200, size=(height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (8 + 6 * k, 10 + 3 * k), 1, (0, 0, 255), -1)
        frames.append(frame)
    return frames


def _lossless_writer(path: Path, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 24.0, size)
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write lossless FFV1 video")
    return writer


def test_rotation_keeps_every_pixel_of_a_small_ball(tmp_path: Path) -> None:
    """A portrait capture (hint 90) must not lose detail when put upright.

    On a real clip the old lossy temporary copy moved pixels by up to 4.7 grey
    levels on average and changed which frames the tracker found the ball in.
    """
    upright_frames = _textured_frames(6)
    height, width = upright_frames[0].shape[:2]
    source = tmp_path / "portrait.mkv"
    writer = _lossless_writer(source, (height, width))
    for frame in upright_frames:
        writer.write(cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
    writer.release()

    restored, size, temporary = upright_video(source, 90)
    try:
        assert size == (width, height)
        capture = cv2.VideoCapture(str(restored))
        read_back = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            read_back.append(frame)
        capture.release()
    finally:
        if temporary:
            restored.unlink(missing_ok=True)

    assert len(read_back) == len(upright_frames)
    for expected, actual in zip(upright_frames, read_back):
        assert np.array_equal(expected, actual)


def test_filled_click_gap_stays_between_the_clicks_around_it() -> None:
    """Filled frames lie on the line between the clicks that bracket the gap.

    A real marking session clicked eight frames moving 24 px a frame down the
    image at 24 fps, skipped seven, then clicked again. The old filler put the
    first skipped frame 34 px *behind* the last click, so the fit saw the ball
    move backwards.
    """
    clicks = {k: (10.0 + 3.0 * k, 24.0 * k) for k in range(8)}
    clicks[15] = (10.0 + 3.0 * 15, 24.0 * 15)
    track = track_from_clicks(clicks, fps=24.0)

    filled = [tp for tp in track if tp.predicted]
    assert [tp.frame_idx for tp in filled] == list(range(8, 15))
    for tp in filled:
        assert tp.u == pytest.approx(10.0 + 3.0 * tp.frame_idx)
        assert tp.v == pytest.approx(24.0 * tp.frame_idx)
    vs = [tp.v for tp in track]
    assert vs == sorted(vs)


# ---------------------------------------------------------------------------
# Results that should not be trusted must say so
# ---------------------------------------------------------------------------


def _good_metrics(**changes: object) -> dict:
    metrics = {
        "release_speed_kmh": 128.0,
        "pitch_length_m": 6.0,
        "pitch_line_m": 0.1,
        "fit_error_px": 2.0,
        "confidence": {"trajectory_fit_ok": True, "within_fit_limits": True},
    }
    metrics.update(changes)
    return metrics


def test_result_checks_stay_quiet_on_a_believable_result() -> None:
    assert result_checks(_good_metrics(), fps=120.0) == []


@pytest.mark.parametrize(
    ("changes", "fps", "expected_start"),
    [
        ({"release_speed_kmh": 199.1}, None, "A release speed of 199 km/h"),
        ({"release_speed_kmh": 29.0}, None, "A release speed of 29 km/h"),
        ({"pitch_line_m": -7.17}, None, "The bounce came out off the pitch"),
        ({"pitch_length_m": 25.0}, None, "The bounce came out off the pitch"),
        (
            {"confidence": {"trajectory_fit_ok": False, "within_fit_limits": True}},
            None,
            "The tracked points don't follow a real ball flight (the fitted path misses",
        ),
        (
            {"confidence": {"trajectory_fit_ok": True, "within_fit_limits": False}},
            None,
            "The fit ran into the edge",
        ),
        ({}, 24.0, "This clip plays at 24 frames per second"),
    ],
)
def test_each_result_check_fires_on_its_own(
    changes: dict, fps: float | None, expected_start: str
) -> None:
    checks = result_checks(_good_metrics(**changes), fps=fps)
    assert len(checks) == 1
    assert checks[0].startswith(expected_start)


def test_older_saved_results_get_checks_without_new_fields() -> None:
    """Records saved before this change have no fit error or limit flag."""
    old = {
        "release_speed_kmh": 29.5,
        "pitch_length_m": 17.8,
        "pitch_line_m": -0.24,
        "confidence": {"trajectory_fit_ok": False},
    }
    checks = delivery_checks({"metrics": old})
    assert checks[0] == POOR_FIT.format(detail="")
    assert any(c.startswith("A release speed of 30 km/h") for c in checks)
    assert delivery_checks({"metrics": old, "checks": []}) == []


def test_fit_stopped_on_a_velocity_limit_is_reported() -> None:
    """A short straight track pins the fit to its slowest allowed release."""
    calib = project_stumps_from_synthetic_camera(default_camera())
    clicks = {k: (620.0 + 3.0 * k, 300.0 + 24.0 * k) for k in range(8)}
    clicks[15] = (665.0, 660.0)
    fit = fit_trajectory(track_from_clicks(clicks, fps=120.0), calib, fps=120.0)
    metrics = metrics_to_dict(compute_metrics(fit))
    assert fit.hit_fit_limits
    assert metrics["confidence"]["within_fit_limits"] is False
    assert AT_FIT_LIMIT in result_checks(metrics)


def test_clean_synthetic_deliveries_raise_no_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """No false alarms on perfect detections (0 of 300 such cases fired on 2026-11-13)."""
    fits = []
    real_fit = benchmark.fit_trajectory

    def keep(*args: object, **kwargs: object):
        fits.append(real_fit(*args, **kwargs))
        return fits[-1]

    monkeypatch.setattr(benchmark, "fit_trajectory", keep)
    config, camera = load_config(), protocol_camera()
    for index in range(12):
        benchmark.run_case(index, benchmark.DEFAULT_SEED, camera, config, mode="noise_free")
    assert len(fits) == 12
    for fit in fits:
        assert result_checks(metrics_to_dict(compute_metrics(fit))) == []


def test_capture_with_a_slow_clock_is_flagged(tmp_path: Path) -> None:
    """The real clips play at 24 fps; the same flight timed at 24 fps must warn.

    The synthetic 120 fps capture passes with no checks. Re-timing its sidecar
    to 24 fps (what a slow-motion export does) makes every speed five times too
    slow, and the result must say so instead of reporting ~25 km/h quietly.
    """
    sidecar_path, _gt = write_synthetic_capture(tmp_path / "clip")
    clean = analyze_capture(sidecar_path, detection_method="hsv")
    assert clean["checks"] == []

    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    n = len(payload["presentationTimestampsUs"])
    payload["captureFps"] = 24
    payload["measuredFps"] = 24.0
    payload["presentationTimestampsUs"] = [round(i * 1_000_000 / 24) for i in range(n)]
    payload["wallClockSeconds"] = payload["durationSeconds"] = (n - 1) / 24
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    slow = analyze_capture(sidecar_path, detection_method="hsv")
    assert slow["metrics"]["release_speed_kmh"] < 50.0
    assert any(c.startswith("A release speed of") for c in slow["checks"])
    assert any(c.startswith("This clip plays at 24 frames") for c in slow["checks"])
