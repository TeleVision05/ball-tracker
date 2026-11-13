"""``balltracker`` command line: ``demo`` and ``benchmark``.

Failures print a plain-language message with what to do next; the technical
details go to a log file next to the output, never to the terminal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # never open a plotting window

log = logging.getLogger("balltracker.cli")

FFMPEG_NOTE = (
    "Note: ffmpeg was not found, so overlay.mp4 is MPEG-4 Part 2, which some browsers "
    "will not play. Install ffmpeg (macOS: brew install ffmpeg; Ubuntu: sudo apt install "
    "ffmpeg) and run the demo again for an H.264 file."
)
CODEC_MESSAGE = (
    "Could not write a video file: this OpenCV build has no MPEG-4 encoder. "
    'Reinstall it with  pip install --force-reinstall "opencv-python>=4.10,<6"  '
    "and run  balltracker demo  again."
)


def _log_to(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    return handler


def _print_demo(summary: dict, out_dir: Path) -> None:
    from .demo import comparison_rows

    clip, calib, track = summary["clip"], summary["calibration"], summary["tracking"]
    print("BallTracker demo: one synthetic delivery, analysed like a real clip\n")
    print(
        f"  1. Simulated the delivery and rendered it: {clip['resolution'][0]}x"
        f"{clip['resolution'][1]}, {clip['fps']} fps, {clip['frames']} frames."
    )
    print(
        f"  2. Calibrated the camera from {calib['click_noise_px']} px-noisy stump clicks: "
        f"{calib['rmse_px']:.2f} px reprojection error."
    )
    print(
        f"  3. Tracked the red ball with the colour detector: {track['points']} of "
        f"{clip['frames_with_ball']} ball frames."
    )
    print(
        "  4. Fitted the flight before and after the bounce: "
        f"{summary['fit_reprojection_rmse_px']:.2f} px reprojection error.\n"
    )
    print(f"  {'':16}{'Estimate':<24}Simulator truth")
    for label, est, tru in comparison_rows(summary["estimate"], summary["truth"]):
        print(f"  {label:<16}{est:<24}{tru}")
    plain = summary["plain_language"]
    print(f"\n  In words: {plain['speed_text']}, {plain['bounce_text'].lower()}.")
    print(
        "  Swing is not estimated yet (the fit has no sideways force), so it reads 0."
        "\n  Synthetic only: see docs/ACCURACY.md for errors over many deliveries."
    )
    files = sorted([*summary["files"].values(), "summary.json"])
    print(f"\nFiles in {out_dir}: " + ", ".join(files))
    print(f"Finished in {summary['runtime_s']:.0f} s.")
    if summary["overlay_codec"] is None:
        print("\n" + FFMPEG_NOTE)
    elif not summary["overlay_is_h264"]:
        print(
            "\nNote: ffmpeg could not convert overlay.mp4 to H.264, so some browsers may "
            "not play it. It still opens in VLC."
        )


def run_demo_command(out_dir: Path, seed: int) -> int:
    """Run the demo and translate failures into plain messages. Returns an exit code."""
    from .demo import run_demo
    from .demo_scene import VideoWriteError

    log_path = out_dir / "demo.log"
    try:
        handler = _log_to(log_path)
    except OSError:
        print(f"Could not write to {out_dir}. Choose another folder with --out.")
        return 1
    try:
        summary = run_demo(out_dir, seed=seed)
    except PermissionError:
        log.exception("demo output folder is not writable")
        print(f"Could not write to {out_dir}. Choose another folder with --out.")
        return 1
    except VideoWriteError:
        log.exception("video writer failed")
        print(CODEC_MESSAGE)
        return 1
    except Exception:
        log.exception("demo failed")
        print(
            "The demo stopped before it finished. Technical details are in "
            f"{log_path}; please open an issue and attach that file."
        )
        return 1
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    _print_demo(summary, out_dir)
    return 0


def run_benchmark_command(n: int, seed: int, out_path: Path) -> int:
    """Run the benchmark, write the markdown report and print the headline table."""
    from .benchmark import METRICS, MODES, error_stats, run_benchmark
    from .benchmark_report import render_accuracy_markdown

    if n < 1:
        print("--n must be at least 1.")
        return 2
    rows_by_mode = {}
    for mode in MODES:
        print(f"Running {n} synthetic deliveries (seed {seed}, {mode.replace('_', ' ')}) ...")
        rows_by_mode[mode] = run_benchmark(n, seed, mode=mode)
    rows = rows_by_mode["pipeline"]
    command = f"balltracker benchmark --n {n} --seed {seed}"
    text = render_accuracy_markdown(rows_by_mode, command, seed, dt.date.today().isoformat())
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    except OSError:
        print(f"Could not write {out_path}. Choose another file with --out.")
        return 1
    stats = error_stats(rows)
    print(f"\n  {'Metric':<26}{'median':>10}{'90th pct':>10}")
    for key, label, unit, _scale in METRICS:
        s = stats[key]
        print(f"  {label + ' (' + unit + ')':<26}{s['median']:>10.1f}{s['p90']:>10.1f}")
    print(f"\nSynthetic only. Wrote {out_path}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .benchmark import DEFAULT_N, DEFAULT_SEED
    from .demo import DEMO_SEED

    parser = argparse.ArgumentParser(
        prog="balltracker",
        description="Cricket bowling analytics from behind-the-stumps video.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run the whole pipeline on a synthetic delivery")
    demo.add_argument("--out", type=Path, default=Path("out/demo"), help="Output folder")
    demo.add_argument("--seed", type=int, default=DEMO_SEED, help="Seed for click noise")
    bench = sub.add_parser("benchmark", help="Measure accuracy on many synthetic deliveries")
    bench.add_argument("--n", type=int, default=DEFAULT_N, help="Number of deliveries")
    bench.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    bench.add_argument(
        "--out", type=Path, default=Path("docs/ACCURACY.md"), help="Markdown report path"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return run_demo_command(args.out, args.seed)
    return run_benchmark_command(args.n, args.seed, args.out)


if __name__ == "__main__":
    sys.exit(main())
