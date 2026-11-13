#!/usr/bin/env python3
"""CLI for analyzing one Android capture sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an Android capture sidecar")
    parser.add_argument(
        "--sidecar",
        type=Path,
        required=True,
        help="Version-2 capture JSON from the phone",
    )
    parser.add_argument("--video", type=Path, default=None, help="Optional path to the clip")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    parser.add_argument(
        "--method",
        choices=["auto", "hsv", "white", "motion"],
        default="auto",
        help="Ball detection method (default: auto)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Config YAML path")
    args = parser.parse_args()

    from balltracker.capture_analysis import analyze_capture

    try:
        result = analyze_capture(
            args.sidecar,
            video_path=args.video,
            detection_method=args.method,
            config_path=args.config,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote analysis to {args.out}")
    summary = result.get("summary")
    if isinstance(summary, dict) and "speed_text" in summary and "bounce_text" in summary:
        print(f"{summary['speed_text']} | {summary['bounce_text']}")
    else:
        metrics = result["metrics"]
        print(
            f"Release speed: {metrics['release_speed_kmh']:.1f} km/h | "
            f"Pitch length: {metrics['pitch_length_m']:.2f} m | "
            f"Line: {metrics['pitch_line_m']:.3f} m"
        )


if __name__ == "__main__":
    main()
