"""Smoke tests for BallTracker analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from balltracker.calib_io import load_calibration, save_calibration
from balltracker.calibration import (
    is_calibration_acceptable,
    project_stumps_from_synthetic_camera,
    reprojection_rmse,
    solve_pnp,
)
from balltracker.flight_window import suggest_flight_window
from balltracker.metrics import compute_metrics, metrics_to_dict
from balltracker.physics import FlightSegment, fit_trajectory, propagate
from balltracker.pipeline import analyze_delivery, evaluate_against_gt
from balltracker.pitch import stump_landmarks_3d
from balltracker.plain_language import (
    bounce_description,
    length_label,
    line_label,
    summarize_delivery,
)
from balltracker.release import ReleaseSeed, mediapipe_available
from balltracker.synthetic import _random_delivery, default_camera, generate_delivery, simulate_trajectory
from balltracker.tracking import TrackPoint, track_delivery_hybrid, track_video


def test_imports() -> None:
    import balltracker.calib_io  # noqa: F401
    import balltracker.calibration  # noqa: F401
    import balltracker.detection  # noqa: F401
    import balltracker.flight_window  # noqa: F401
    import balltracker.metrics  # noqa: F401
    import balltracker.physics  # noqa: F401
    import balltracker.pipeline  # noqa: F401
    import balltracker.plain_language  # noqa: F401
    import balltracker.release  # noqa: F401
    import balltracker.tracking  # noqa: F401


def test_pnp_recovers_synthetic_camera() -> None:
    cam = default_camera()
    landmarks = stump_landmarks_3d()
    pts3 = np.stack(list(landmarks.values()))
    uv = cam.project(pts3)
    img_pts = {
        k: uv[i]
        for i, k in enumerate(landmarks.keys())
        if np.all(np.isfinite(uv[i]))
    }
    K = cam.intrinsic_matrix()
    calib = solve_pnp(img_pts, image_size=cam.resolution, K=K)
    rmse = reprojection_rmse(calib, img_pts, landmarks)
    assert rmse < 2.0, f"PnP RMSE too high: {rmse:.3f}px"
    assert is_calibration_acceptable(rmse, max_rmse=2.0)


def test_calib_io_roundtrip(tmp_path: Path) -> None:
    cam = default_camera()
    calib = project_stumps_from_synthetic_camera(cam)
    landmarks = stump_landmarks_3d()
    pts3 = np.stack(list(landmarks.values()))
    uv = cam.project(pts3)
    stump_pts = {
        k: uv[i]
        for i, k in enumerate(landmarks.keys())
        if np.all(np.isfinite(uv[i]))
    }

    calib_path = tmp_path / "test.calib.json"
    save_calibration(calib, calib_path, stump_image_points=stump_pts)
    loaded, loaded_stumps = load_calibration(calib_path)

    assert loaded.resolution == calib.resolution
    assert np.allclose(loaded.K, calib.K)
    assert np.allclose(loaded.R, calib.R)
    assert np.allclose(loaded.t, calib.t)
    assert loaded.rms_reprojection_error == pytest.approx(calib.rms_reprojection_error)
    assert set(loaded_stumps.keys()) == set(stump_pts.keys())


def test_physics_propagate_roundtrip() -> None:
    x0 = np.array([0.0, 1.0, 2.0])
    v0 = np.array([0.1, 30.0, -1.0])
    seg = FlightSegment(t0=0.0, x0=x0, v0=v0, drag_coeff=0.0)
    times = np.linspace(0, 0.5, 20)
    pos = propagate(seg, times)
    assert pos.shape == (20, 3)
    assert pos[0, 2] == pytest.approx(2.0)
    assert pos[-1, 2] < pos[0, 2]


def test_metrics_dict_keys() -> None:
    cam = default_camera()
    calib = project_stumps_from_synthetic_camera(cam)
    rng = np.random.default_rng(0)
    params = _random_delivery(rng)
    times, positions, _bounce_idx, _bounce_pos = simulate_trajectory(params, dt=1 / 120.0)
    track = [
        TrackPoint(
            frame_idx=i,
            t_s=float(times[i]),
            u=float(uv[0]),
            v=float(uv[1]),
            radius_px=6.0,
            confidence=1.0,
            predicted=False,
        )
        for i, uv in enumerate(cam.project(positions))
        if np.all(np.isfinite(uv))
    ]
    fit = fit_trajectory(track, calib, fps=120.0)
    metrics = compute_metrics(fit)
    d = metrics_to_dict(metrics)
    for key in (
        "release_speed_kmh",
        "pitch_line_m",
        "pitch_length_m",
        "beta_s_deg",
        "c_s",
        "lateral_drift_m",
        "turn_deg",
        "turn_m",
        "bounce_y_m",
        "confidence",
    ):
        assert key in d


def test_resolve_calibration_requires_calib(tmp_path: Path) -> None:
    fake_video = tmp_path / "clip.mp4"
    fake_video.write_bytes(b"not a real video")
    with pytest.raises(RuntimeError, match="calibrate_stumps"):
        analyze_delivery(fake_video)


@pytest.fixture
def synthetic_delivery(tmp_path: Path) -> tuple[Path, Path]:
    """Generate one tiny synthetic delivery in a temp dir (no committed demo data)."""
    out_dir = tmp_path / "synthetic"
    out_dir.mkdir()
    rng = np.random.default_rng(42)
    generate_delivery("delivery_0001", out_dir, fps=120, rng=rng)
    video = out_dir / "delivery_0001.mp4"
    gt_path = out_dir / "delivery_0001.json"
    assert video.exists()
    assert gt_path.exists()
    return video, gt_path


def test_track_synthetic_video(synthetic_delivery: tuple[Path, Path]) -> None:
    video, _ = synthetic_delivery
    track = track_video(video, fps=120)
    assert len(track) > 10
    detected = sum(1 for tp in track if not tp.predicted)
    assert detected > len(track) * 0.5


def test_pipeline_end_to_end(synthetic_delivery: tuple[Path, Path], tmp_path: Path) -> None:
    video, gt_path = synthetic_delivery
    calib_path = tmp_path / "session.calib.json"

    with gt_path.open() as f:
        gt = json.load(f)
    cam = default_camera(fps=int(gt["fps"]))
    calib = project_stumps_from_synthetic_camera(cam)
    landmarks = stump_landmarks_3d()
    pts3 = np.stack(list(landmarks.values()))
    uv = cam.project(pts3)
    stump_pts = {
        k: uv[i]
        for i, k in enumerate(landmarks.keys())
        if np.all(np.isfinite(uv[i]))
    }
    save_calibration(calib, calib_path, stump_image_points=stump_pts)

    result = analyze_delivery(video, calib_path=calib_path, use_companion_track=True)
    assert "metrics" in result
    assert "track" in result
    assert result["calib_rmse_px"] < 2.0

    errors = evaluate_against_gt(result["metrics"], gt_path)
    assert "release_speed_kmh_abs_err" in errors


def test_release_speed_vs_gt(synthetic_delivery: tuple[Path, Path], tmp_path: Path) -> None:
    video, gt_path = synthetic_delivery
    calib_path = tmp_path / "session.calib.json"

    with gt_path.open() as f:
        gt = json.load(f)
    cam = default_camera(fps=int(gt["fps"]))
    calib = project_stumps_from_synthetic_camera(cam)
    save_calibration(calib, calib_path)

    result = analyze_delivery(video, calib_path=calib_path, use_companion_track=True)
    est = result["metrics"]["release_speed_kmh"]
    gt_speed = gt["radar_speed_kmh"]
    rel_err = abs(est - gt_speed) / gt_speed
    assert rel_err <= 0.15, f"Release speed error {rel_err:.1%} (est={est:.1f}, gt={gt_speed:.1f})"


def test_bounce_position_vs_gt(synthetic_delivery: tuple[Path, Path], tmp_path: Path) -> None:
    video, gt_path = synthetic_delivery
    calib_path = tmp_path / "session.calib.json"

    with gt_path.open() as f:
        gt = json.load(f)
    cam = default_camera(fps=int(gt["fps"]))
    calib = project_stumps_from_synthetic_camera(cam)
    save_calibration(calib, calib_path)

    result = analyze_delivery(video, calib_path=calib_path, use_companion_track=True)
    gt_bounce = np.array(gt["bounce_position_m"])
    est_line = result["metrics"]["pitch_line_m"]
    est_y = result["metrics"]["bounce_y_m"]
    err = np.linalg.norm([est_line - gt_bounce[0], est_y - gt_bounce[1]])
    assert err <= 1.5, f"Bounce position error {err:.2f}m exceeds 1.5m"


def test_plain_language_bounce() -> None:
    assert length_label(1.0) == "yorker"
    assert length_label(3.0) == "full"
    assert length_label(5.0) == "good length"
    assert length_label(8.0) == "short"
    assert length_label(12.0) == "too short"

    assert line_label(0.02) == "middle stump"
    assert line_label(0.5) == "off stump"
    assert line_label(-0.5) == "leg stump"

    assert bounce_description(0.5, 5.0) == "Good length, toward off stump"
    assert bounce_description(-0.5, 1.0) == "Yorker, toward leg stump"
    assert bounce_description(0.02, 5.0) == "Good length, on the stumps"

    summary = summarize_delivery(
        {"release_speed_kmh": 132.4, "pitch_line_m": 0.3, "pitch_length_m": 6.0}
    )
    for key in ("speed_kmh", "speed_text", "bounce_text", "length", "line"):
        assert key in summary
    assert summary["length"] == "good length"
    assert summary["line"] == "off stump"
    assert "132.4" in summary["speed_text"]

    empty_summary = summarize_delivery({})
    assert empty_summary["speed_kmh"] is None
    assert empty_summary["length"] == "unknown"


def test_suggest_flight_window_synthetic(synthetic_delivery: tuple[Path, Path]) -> None:
    video, gt_path = synthetic_delivery
    with gt_path.open() as f:
        gt = json.load(f)
    total_frames = int(gt["frame_count"])

    start, end = suggest_flight_window(video)
    assert 0 <= start < end
    assert end < total_frames


def test_track_delivery_hybrid_with_mock_seed(
    synthetic_delivery: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    video, gt_path = synthetic_delivery
    with gt_path.open() as f:
        gt = json.load(f)
    fps = float(gt["fps"])
    track_2d = gt["track_2d_px"]

    p0 = track_2d[0]
    p1 = track_2d[min(5, len(track_2d) - 1)]
    du, dv = p1[0] - p0[0], p1[1] - p0[1]
    norm = float(np.hypot(du, dv)) or 1.0
    seed = ReleaseSeed(
        frame_idx=0,
        u=float(p0[0]),
        v=float(p0[1]),
        direction_uv=(du / norm, dv / norm),
        confidence=0.8,
    )

    monkeypatch.setattr("balltracker.tracking.estimate_release_seed", lambda *a, **k: seed)

    track = track_delivery_hybrid(video, fps=fps)
    assert len(track) > 5


def test_mediapipe_available_bool() -> None:
    assert isinstance(mediapipe_available(), bool)
