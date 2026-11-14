"""Fast synthetic accuracy check: a regression in the fit or the metrics fails CI.

Runs a small, seeded version of ``balltracker benchmark`` (16 deliveries, so the
measured values differ from the 500-case report). Bounds are roughly two to three
times the 16-case errors measured on 2026-11-13, so ordinary changes pass
and a broken fit does not. Synthetic only: see docs/ACCURACY.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from balltracker.benchmark import DEFAULT_SEED, METRICS, error_stats, run_benchmark
from balltracker.benchmark_report import render_accuracy_markdown
from balltracker.scenario import DeliverySpec, simulate_delivery

N_CASES = 16

# metric -> (max median abs error, max 90th percentile abs error), in report units.
PIPELINE_BOUNDS = {
    "release_speed_kmh": (12.0, 30.0),
    "bounce_line_m": (4.5, 10.0),
    "bounce_length_m": (150.0, 350.0),
    "turn_deg": (2.5, 7.0),
}
NOISE_FREE_BOUNDS = {
    "release_speed_kmh": (6.0, 15.0),
    "bounce_line_m": (0.6, 1.8),
    "bounce_length_m": (35.0, 120.0),
    "turn_deg": (1.2, 3.0),
}


@pytest.fixture(scope="module")
def rows_by_mode() -> dict[str, list[dict]]:
    return {
        mode: run_benchmark(N_CASES, DEFAULT_SEED, mode=mode)
        for mode in ("pipeline", "no_filter", "noise_free")
    }


def _assert_within(rows: list[dict], bounds: dict[str, tuple[float, float]]) -> None:
    assert not [r for r in rows if "failed" in r], "the fit failed on a synthetic delivery"
    stats = error_stats(rows)
    for key, (max_median, max_p90) in bounds.items():
        assert stats[key]["median"] <= max_median, (key, stats[key])
        assert stats[key]["p90"] <= max_p90, (key, stats[key])


def test_noisy_pipeline_errors_stay_bounded(rows_by_mode) -> None:
    _assert_within(rows_by_mode["pipeline"], PIPELINE_BOUNDS)


def test_noise_free_fit_errors_stay_bounded(rows_by_mode) -> None:
    _assert_within(rows_by_mode["noise_free"], NOISE_FREE_BOUNDS)


def test_benchmark_is_reproducible() -> None:
    first = run_benchmark(3, DEFAULT_SEED)
    again = run_benchmark(3, DEFAULT_SEED)
    assert [r["estimate"] for r in first] == [r["estimate"] for r in again]


def test_aimed_delivery_pitches_where_asked() -> None:
    delivery = simulate_delivery(DeliverySpec(speed_kmh=130.0, line_m=0.1, length_m=5.0))
    truth = delivery.truth
    assert truth["release_speed_kmh"] == pytest.approx(130.0, abs=0.01)
    assert truth["bounce_length_m"] == pytest.approx(5.0, abs=0.1)
    assert truth["bounce_line_m"] == pytest.approx(0.1, abs=0.02)
    assert truth["swing_deg"] == pytest.approx(0.0, abs=1e-6)


def test_report_is_labelled_synthetic_and_reproducible(rows_by_mode) -> None:
    text = render_accuracy_markdown(rows_by_mode, "balltracker benchmark --n 16", 7, "2026-11-13")
    assert text.startswith("# Accuracy: synthetic deliveries only")
    assert "synthetic-only validation" in text
    assert "`balltracker benchmark --n 16`" in text
    assert "Seed: `7`" in text
    for _key, label, _unit, _scale in METRICS:
        assert f"| {label} |" in text


def test_error_stats_handles_no_rows() -> None:
    stats = error_stats([])
    assert all(np.isnan(s["median"]) for s in stats.values())
