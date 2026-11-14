"""Write docs/ACCURACY.md from benchmark rows."""

from __future__ import annotations

from .benchmark import BOWLING_STYLES, FRAME_RATES, METRICS, error_stats
from .landmarks import WARN_RMSE_PX

_HEADER = """# Accuracy: synthetic deliveries only

> **This is synthetic-only validation.** Every number on this page compares
> BallTracker against its own physics simulator, not against real deliveries.
> No real delivery has been measured against radar or a marked bounce yet
> (see [GROUND_TRUTH.md](GROUND_TRUTH.md) for how that will be done).

**What this proves.** Given ball positions of the quality a phone detector
produces, it measures how well the tracker's outlier filter, the stump
calibration, the bounce-segmented fit and the metric formulas recover the
simulator's speed, bounce spot and turn, and how that changes with frame rate
and bowling style. The fast test in `tests/test_benchmark.py` re-runs a small
version of this and fails if the fit gets worse.

**What this does not prove.** It says nothing about real footage. The image
detector is not run here: detections are simulated as true pixel positions plus
noise (`balltracker demo` runs the real detector on a rendered clip). The
simulator shares the fit's assumptions: a pinhole camera with no lens
distortion and a known field of view, a flat pitch, no air drag, and a bounce
model from the same family as the one the fit expects. Real balls, lenses,
pitches and lighting will add error that this page cannot see.
"""


def _fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.1f}"


def _main_table(stats: dict[str, dict]) -> list[str]:
    """``stats`` maps mode to ``error_stats`` output."""
    lines = [
        "| Metric | Unit | Median error | 90th percentile | Median signed error | "
        "Median error, filter skipped | Median error, noise-free |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, label, unit, _scale in METRICS:
        s = stats["pipeline"][key]
        lines.append(
            f"| {label} | {unit} | {_fmt(s['median'])} | {_fmt(s['p90'])} | "
            f"{_fmt(s['bias'])} | {_fmt(stats['no_filter'][key]['median'])} | "
            f"{_fmt(stats['noise_free'][key]['median'])} |"
        )
    return lines


def _breakdown(rows: list[dict], field: str, title: str, values) -> list[str]:
    header = f"| {title} | Cases | "
    header += " | ".join(f"{label} ({unit})" for _k, label, unit, _s in METRICS) + " |"
    lines = [header, "|---|---:|" + "---:|" * len(METRICS)]
    for value in values:
        subset = [r for r in rows if r[field] == value]
        stats = error_stats(subset)
        cells = " | ".join(_fmt(stats[key]["median"]) for key, *_ in METRICS)
        lines.append(f"| {value} | {len(subset)} | {cells} |")
    return lines


def _setup_section() -> list[str]:
    styles = ", ".join(
        f"{name} {int(r['speed'][0])} to {int(r['speed'][1])} km/h ({int(r['weight'] * 100)}%)"
        for name, r in BOWLING_STYLES.items()
    )
    rates = ", ".join(str(f) for f in FRAME_RATES)
    return [
        "## How each case is built",
        "",
        f"- **Delivery.** Bowling style: {styles}. Length in front of the batter's "
        "stumps 1.5 to 10 m (medium 1.5 to 9 m, spin 1 to 6 m), line 35 cm leg side to "
        "45 cm off side, swing "
        "up to 3 m/s² sideways (spin 0.8), a sideways kick at the bounce of up to "
        "2.7 m/s for spin and up to 0.45 m/s for seam, release 0.8 to 1.4 m in front of "
        "the bowler's stumps and up to 45 cm either side of them.",
        "- **Flight.** `synthetic.simulate_trajectory` at 1 ms steps: gravity plus the "
        "constant swing force, a bounce with random restitution (0.65 to 0.85) and "
        "friction (0.05 to 0.25), then flight to 1 m past the batter's stumps.",
        "- **Camera.** Behind the bowler: 4 m back, 1.66 m high, 55° field of view, "
        "1280x720, tilted so both stump sets are fully in frame as "
        "[CAPTURE_PROTOCOL.md](CAPTURE_PROTOCOL.md) asks.",
        f"- **Capture.** Frame rate chosen from {rates} fps with a random phase. "
        "Detection jitter 0.5 to 2 px (Gaussian), 0 to 15% of frames missed, half the "
        "cases also lose a burst of 1 to 4 frames in the second half of the flight, the "
        "first 0 to 3 frames after release are hidden, and 2% of detections are strays "
        "5 to 15 px off.",
        "- **Tracking.** The detections go through `tracking.filter_ballistic_segment`, "
        "the outlier filter every video tracking method ends with.",
        "- **Calibration.** PnP on the stump points visible in the frame, as a person "
        "would click them, each off by 1.5 px (Gaussian), with the configured field of "
        "view.",
        "- **Measurement.** `physics.fit_trajectory` then `metrics.compute_metrics`, "
        "with the physics settings from `config/default.yaml`, as the video pipeline "
        "calls them.",
        "- **Filter-skipped column.** The same noisy detections sent straight to the fit.",
        "- **Noise-free column.** The same deliveries and frame rates with perfect "
        "detections, perfect clicks and no filter: the error of the fit model alone.",
        "",
    ]


def _frame_rate_sentence(rows: list[dict]) -> str:
    slow = error_stats([r for r in rows if r["fps"] == min(FRAME_RATES)])
    fast = error_stats([r for r in rows if r["fps"] == max(FRAME_RATES)])
    # Swing is left out: its estimate is constant, so its "error" only reflects the sample.
    measured = [(key, label) for key, label, _u, _s in METRICS if key != "swing_deg"]
    better = [label for key, label in measured if fast[key]["median"] < slow[key]["median"]]
    worse = [label for _key, label in measured if label not in better]
    text = (
        f"- **Frame rate.** Going from {min(FRAME_RATES)} to {max(FRAME_RATES)} fps lowers "
        f"the median error for: {', '.join(better).lower() or 'none of the metrics'}."
    )
    if worse:
        text += f" It does not for: {', '.join(worse).lower()}."
    return text


def _swing_sentence(rows: list[dict], swing: dict) -> str:
    estimates = {r["estimate"]["swing_deg"] for r in rows if "estimate" in r}
    if estimates != {0.0}:
        return "- **Swing.** The fit estimates swing; see the table."
    return (
        "- **Swing is not measured yet.** The pre-bounce fit has gravity and no "
        "sideways force, so its swing estimate is 0° in every case. The swing error "
        f"above is the size of the simulated swing (median {_fmt(swing['median'])}°). "
        "The missing swing also leaks into the turn estimate, because turn is measured "
        "against the fitted straight approach."
    )


def _filter_sentence(stats: dict[str, dict]) -> str:
    piped, skipped = stats["pipeline"], stats["no_filter"]
    keys = ("release_speed_kmh", "bounce_length_m")
    if all(skipped[k]["median"] < piped[k]["median"] for k in keys):
        return (
            "- **The tracker's outlier filter is the largest error source.** Sending the "
            "same detections straight to the fit lowers the median speed error from "
            f"{_fmt(piped['release_speed_kmh']['median'])} to "
            f"{_fmt(skipped['release_speed_kmh']['median'])} km/h and the median length "
            f"error from {_fmt(piped['bounce_length_m']['median'])} to "
            f"{_fmt(skipped['bounce_length_m']['median'])} cm. The filter fits one "
            "quadratic through the whole track, bounce included, so the frames around "
            "the bounce look like outliers and are dropped, and the fit then places the "
            "bounce from the frames either side of the gap."
        )
    return "- **Outlier filter.** Skipping the tracker's outlier filter does not help."


def _findings(rows: list[dict], stats: dict[str, dict]) -> list[str]:
    piped = stats["pipeline"]
    length, line = piped["bounce_length_m"], piped["bounce_line_m"]
    return [
        "## What the numbers say",
        "",
        _filter_sentence(stats),
        _swing_sentence(rows, piped["swing_deg"]),
        "- **Length is the weakest bounce coordinate.** Looking down the pitch, a few "
        "pixels near the bounce move the bounce point a long way along the pitch: "
        f"median length error {_fmt(length['median'])} cm against "
        f"{_fmt(line['median'])} cm for line.",
        _frame_rate_sentence(rows),
        "",
    ]


def render_accuracy_markdown(
    rows_by_mode: dict[str, list[dict]], command: str, seed: int, run_date: str
) -> str:
    """Build the full ACCURACY.md text from rows for every mode in ``benchmark.MODES``."""
    rows = rows_by_mode["pipeline"]
    stats = {mode: error_stats(mode_rows) for mode, mode_rows in rows_by_mode.items()}
    failed = sum(1 for r in rows if "failed" in r)
    reclicked = sum(1 for r in rows if r["calib_attempts"] > 1)
    still_bad = sum(1 for r in rows if r["calib_rmse_px"] > WARN_RMSE_PX)
    lines = [_HEADER, "## How this page was made", ""]
    lines += [
        f"- Command: `{command}`",
        f"- Seed: `{seed}`",
        f"- Date: {run_date}",
        f"- Cases: {len(rows)} ({failed} where the fit could not run)",
        f"- Calibrations that needed fresh clicks: {reclicked} of {len(rows)} (the app "
        f"asks for new clicks above {WARN_RMSE_PX:.0f} px); {still_bad} still exceeded "
        "it after five rounds.",
        "",
        "Re-running the command with the same seed reproduces these numbers.",
        "",
        "## Errors",
        "",
        "Absolute error of each estimate against the simulator, over all cases. The "
        "first three columns are the full path a video takes after detection.",
        "",
    ]
    lines += _main_table(stats)
    lines += ["", "### Median error by frame rate", ""]
    lines += _breakdown(rows, "fps", "Frame rate (fps)", FRAME_RATES)
    lines += ["", "### Median error by bowling style", ""]
    lines += _breakdown(rows, "style", "Bowling style", list(BOWLING_STYLES))
    lines += [""] + _findings(rows, stats) + _setup_section()
    return "\n".join(lines).rstrip() + "\n"
