"""Characterisation tests for the tracker's pure helpers (no video).

Hand-built inputs pin association, the ballistic filter and manual-click
tracking exactly, so a module split cannot change them silently.
"""

from __future__ import annotations

import pytest

import balltracker.tracking as tracking
from balltracker.detection import BallDetection
from balltracker.tracking import (
    TrackPoint,
    associate_detections,
    filter_ballistic_segment,
    predict_ball_uv,
    track_from_clicks,
)


def _line(n: int, step_u: float = 10.0, step_v: float = 5.0) -> list[BallDetection | None]:
    return [BallDetection(k, 100.0 + step_u * k, 50.0 + step_v * k, 6.0, 0.8) for k in range(n)]


def _points(coords: list[tuple[int, float, float]], predicted: bool = False) -> list[TrackPoint]:
    return [TrackPoint(f, f / 120.0, u, v, 5.0, 0.9, predicted) for f, u, v in coords]


# ---------------------------------------------------------------------------
# associate_detections
# ---------------------------------------------------------------------------


def test_clean_detections_pass_straight_through() -> None:
    track = associate_detections(_line(10), fps=100.0)
    assert [(tp.frame_idx, tp.u, tp.v) for tp in track] == [
        (k, 100.0 + 10 * k, 50.0 + 5 * k) for k in range(10)
    ]
    assert all(not tp.predicted and tp.confidence == 0.8 for tp in track)
    assert [tp.t_s for tp in track] == pytest.approx([k / 100.0 for k in range(10)])


def test_an_outlier_is_replaced_by_a_prediction() -> None:
    dets = _line(10)
    dets[5] = BallDetection(5, 900.0, 600.0, 9.0, 0.9)
    track = associate_detections(dets, fps=100.0)
    assert len(track) == 10
    replaced = track[5]
    assert replaced.predicted and replaced.confidence == 0.2
    assert replaced.radius_px == 6.0, "radius carries over from the previous point"
    # The constant-velocity filter lags a fresh track, so the guess trails the
    # true (150, 75); pinned to today's value.
    assert (replaced.u, replaced.v) == pytest.approx((125.0446, 62.5223), abs=1e-3)
    assert all(not tp.predicted for i, tp in enumerate(track) if i != 5)


def test_missing_frames_are_predicted_up_to_max_gap_then_dropped() -> None:
    dets: list[BallDetection | None] = [None, None] + _line(6)[2:] + [None] * 10
    track = associate_detections(dets, fps=100.0, max_gap=4)
    assert [tp.frame_idx for tp in track] == list(range(2, 10))
    assert [tp.predicted for tp in track] == [False] * 4 + [True] * 4
    assert {tp.confidence for tp in track[4:]} == {0.25}


def test_the_early_jump_gate_admits_a_fast_start() -> None:
    # The Kalman velocity lags a fast start, so residuals grow; the early gate
    # buys a few frames, limited by early_count.
    fast = _line(8, step_u=40.0, step_v=0.0)
    strict = associate_detections(fast, fps=100.0, max_jump_px=30.0)
    assert [tp.predicted for tp in strict] == [False, True, True, True, True]
    relaxed = associate_detections(fast, fps=100.0, max_jump_px=30.0, early_max_jump_px=60.0)
    assert [tp.predicted for tp in relaxed] == [False] * 3 + [True] * 4
    capped = associate_detections(
        fast, fps=100.0, max_jump_px=30.0, early_max_jump_px=60.0, early_count=2
    )
    assert [tp.predicted for tp in capped] == [False] * 2 + [True] * 4


def test_zero_fps_does_not_divide_by_zero() -> None:
    track = associate_detections(_line(3), fps=0.0)
    assert [tp.t_s for tp in track] == pytest.approx([0.0, 1e6, 2e6])


# ---------------------------------------------------------------------------
# filter_ballistic_segment
# ---------------------------------------------------------------------------


def test_short_tracks_are_returned_unchanged() -> None:
    short = _points([(k, 10.0 * k, 0.0) for k in range(5)])
    assert filter_ballistic_segment(short) is short
    mostly_predicted = _points([(k, 10.0 * k, 0.0) for k in range(4)]) + _points(
        [(k, 10.0 * k, 0.0) for k in range(4, 10)], predicted=True
    )
    assert filter_ballistic_segment(mostly_predicted) is mostly_predicted


def test_the_longest_moving_segment_wins() -> None:
    first = [(k, 5.0 * k, 0.0) for k in range(6)]
    second = [(k, 200.0 + 8.0 * k, 0.0) for k in range(6, 16)]
    result = filter_ballistic_segment(_points(first + second))
    assert [tp.frame_idx for tp in result] == list(range(6, 16))


def test_frame_gaps_split_segments_and_predictions_are_dropped() -> None:
    coords = [(k, 5.0 * k, 0.0) for k in range(7)] + [(k, 5.0 * k, 0.0) for k in range(13, 20)]
    track = _points(coords) + _points([(30, 0.0, 0.0)], predicted=True)
    result = filter_ballistic_segment(track)
    assert [tp.frame_idx for tp in result] == list(range(7))


def test_a_stationary_segment_loses_to_a_moving_one() -> None:
    still = [(k, 100.0, 100.0 + 0.5 * k) for k in range(20)]
    moving = [(k, 400.0 + 6.0 * k, 100.0) for k in range(26, 32)]
    result = filter_ballistic_segment(_points(still + moving))
    assert [tp.frame_idx for tp in result] == list(range(26, 32))


def test_a_residual_outlier_is_removed_by_the_quadratic_fit() -> None:
    coords = [(k, 100.0 + 8.0 * k, 50.0 + 0.4 * k * k) for k in range(12)]
    coords[6] = (6, coords[6][1] + 40.0, coords[6][2])
    result = filter_ballistic_segment(_points(coords))
    assert [tp.frame_idx for tp in result] == [k for k in range(12) if k != 6]


def test_no_long_enough_segment_returns_the_measured_points() -> None:
    coords = [(3 * k, 100.0 * k, 0.0) for k in range(8)]
    track = _points(coords) + _points([(50, 0.0, 0.0)], predicted=True)
    result = filter_ballistic_segment(track)
    assert [tp.frame_idx for tp in result] == [3 * k for k in range(8)]


# ---------------------------------------------------------------------------
# Manual clicks
# ---------------------------------------------------------------------------


def test_predict_ball_uv_cases() -> None:
    assert predict_ball_uv({}, 3, 120.0) is None
    one = {5: (10.0, 20.0)}
    assert predict_ball_uv(one, 2, 120.0) == (10.0, 20.0)
    assert predict_ball_uv(one, 5, 120.0) == (10.0, 20.0)
    assert predict_ball_uv(one, 8, 120.0) == (10.0, 20.0)
    two = {0: (0.0, 0.0), 2: (10.0, 4.0)}
    assert predict_ball_uv(two, 5, 120.0) == (25.0, 10.0)
    assert predict_ball_uv(two, 1, 120.0) == (0.0, 0.0)


def test_track_from_clicks_without_filling() -> None:
    assert track_from_clicks({}, fps=120.0) == []
    track = track_from_clicks(
        {4: (1.0, 2.0), 1: (3.0, 4.0)}, fps=100.0, fill_gaps=False, radius_px=4.0
    )
    assert [(tp.frame_idx, tp.u, tp.v, tp.radius_px) for tp in track] == [
        (1, 3.0, 4.0, 4.0),
        (4, 1.0, 2.0, 4.0),
    ]
    assert all(tp.confidence == 1.0 and not tp.predicted for tp in track)


def test_track_from_clicks_fills_short_gaps_only() -> None:
    track = track_from_clicks(
        {0: (0.0, 0.0), 3: (30.0, 0.0), 13: (90.0, 0.0)}, fps=100.0, max_gap=4
    )
    assert [tp.frame_idx for tp in track] == [0, 1, 2, 3, 4, 5, 6, 7, 13]
    assert [tp.predicted for tp in track] == [
        False,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert {tp.confidence for tp in track if tp.predicted} == {0.35}
    assert track[-1].t_s == pytest.approx(0.13)


# ---------------------------------------------------------------------------
# Public import paths
# ---------------------------------------------------------------------------


def test_old_import_paths_still_work() -> None:
    for name in (
        "TrackPoint",
        "associate_detections",
        "filter_ballistic_segment",
        "track_delivery_hybrid",
        "track_video",
        "predict_ball_uv",
        "track_from_clicks",
        "estimate_release_seed",
        "_Kalman2D",
        "_create_csrt_tracker",
    ):
        assert hasattr(tracking, name), name
