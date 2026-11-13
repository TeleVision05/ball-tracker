#!/usr/bin/env python3
"""Interactive stump calibration for real delivery videos (OpenCV desktop)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from balltracker.landmarks import (
    LANDMARK_ORDER,
    MIN_LANDMARKS,
    WARN_RMSE_PX,
    calibrate_from_points,
)


def _print_instructions() -> None:
    print("Click stump landmarks in this order (left-click):")
    for i, name in enumerate(LANDMARK_ORDER, start=1):
        print(f"  {i:2d}. {name}")
    print(f"\nKeys: u=undo  q=quit  Enter/d=done (need >={MIN_LANDMARKS} points)")
    print("Tip: you can also calibrate in the Streamlit UI.")


def _load_points_json(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}


def _read_frame(video_path: Path, frame_idx: int) -> tuple[np.ndarray, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame, (w, h)


def _interactive_points(frame: np.ndarray) -> dict[str, np.ndarray] | None:
    points: dict[str, np.ndarray] = {}
    click_idx = 0
    display = frame.copy()

    def _redraw() -> None:
        nonlocal display
        display = frame.copy()
        for i, (name, uv) in enumerate(points.items()):
            u, v = int(uv[0]), int(uv[1])
            cv2.drawMarker(display, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(
                display,
                f"{i + 1}:{name}",
                (u + 6, v - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if click_idx < len(LANDMARK_ORDER):
            cv2.putText(
                display,
                f"Next: {LANDMARK_ORDER[click_idx]}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            display,
            f"Points: {len(points)}/12  (u=undo q=quit Enter/d=done)",
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _on_mouse(event: int, x: int, y: int, _flags: int, _userdata) -> None:
        nonlocal click_idx
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if click_idx >= len(LANDMARK_ORDER):
            return
        name = LANDMARK_ORDER[click_idx]
        points[name] = np.array([float(x), float(y)], dtype=np.float64)
        click_idx += 1
        _redraw()

    win = "BallTracker stump calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _on_mouse)
    _redraw()

    done = False
    cancelled = False
    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cancelled = True
            break
        if key == ord("u") and points:
            removed = LANDMARK_ORDER[click_idx - 1]
            del points[removed]
            click_idx -= 1
            _redraw()
        if key in (13, ord("d")) and len(points) >= MIN_LANDMARKS:
            done = True
            break

    cv2.destroyAllWindows()
    if cancelled or not done:
        return None
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate camera from stump landmarks")
    parser.add_argument("--video", type=Path, required=True, help="Delivery video path")
    parser.add_argument("--out", type=Path, required=True, help="Output .calib.json path")
    parser.add_argument("--frame", type=int, default=0, help="Frame index for calibration")
    parser.add_argument("--fov", type=float, default=55.0, help="Horizontal FOV (degrees)")
    parser.add_argument(
        "--points",
        type=Path,
        default=None,
        help="Pre-recorded landmark JSON {name: [u,v]} (headless)",
    )
    args = parser.parse_args()

    try:
        frame, image_size = _read_frame(args.video, args.frame)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.points is not None:
        image_points = _load_points_json(args.points)
    else:
        _print_instructions()
        try:
            image_points = _interactive_points(frame)
        except cv2.error as exc:
            print(
                f"OpenCV GUI unavailable ({exc}). Use the Streamlit UI or --points points.json.",
                file=sys.stderr,
            )
            return 1
        if image_points is None:
            print("Calibration cancelled.", file=sys.stderr)
            return 1

    try:
        _calib, rmse = calibrate_from_points(
            image_points, image_size, fov_deg=args.fov, out_path=args.out
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Calibration RMSE: {rmse:.2f} px")
    if rmse > WARN_RMSE_PX:
        print(f"Warning: RMSE {rmse:.2f} px exceeds {WARN_RMSE_PX:.0f} px target.")
    print(f"Saved calibration to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
