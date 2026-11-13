#!/usr/bin/env python3
"""CLI for single-delivery analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a cricket delivery video")
    parser.add_argument("--video", type=Path, required=True, help="Path to delivery MP4")
    parser.add_argument(
        "--calib",
        type=Path,
        default=None,
        help="Camera calibration JSON from calibrate_stumps.py",
    )
    parser.add_argument("--gt", type=Path, default=None, help="Ground-truth JSON for evaluation")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML path")
    parser.add_argument(
        "--method",
        choices=["auto", "hsv", "white", "motion"],
        default="auto",
        help="Ball detection method (default: auto)",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to analyze")
    parser.add_argument("--end-frame", type=int, default=None, help="Last frame to analyze")
    parser.add_argument(
        "--use-companion-track",
        action="store_true",
        help="Use track_2d_px from companion JSON instead of detector",
    )
    args = parser.parse_args()

    from balltracker.pipeline import analyze_delivery, evaluate_against_gt

    try:
        result = analyze_delivery(
            args.video,
            calib_path=args.calib,
            config_path=args.config,
            detection_method=args.method,
            use_companion_track=args.use_companion_track,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.calib is None:
            print(
                "Provide --calib path.calib.json or place session.calib.json "
                "next to the video folder.",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc

    if args.gt is not None:
        result["evaluation"] = evaluate_against_gt(result["metrics"], args.gt)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote analysis to {args.out}")
    m = result["metrics"]
    print(
        f"Release speed: {m['release_speed_kmh']:.1f} km/h | "
        f"Pitch length: {m['pitch_length_m']:.2f} m | "
        f"Line: {m['pitch_line_m']:.3f} m"
    )
    for check in result.get("checks", []):
        print(f"Check this result: {check}")
    if "evaluation" in result:
        print("GT errors:", result["evaluation"])


if __name__ == "__main__":
    main()
