"""Export detections/analysis as JSON, CSV, or a trajectory-burned-in MP4.

Video export uses OpenCV to decode frames and draw the trajectory overlay,
then (when available) shells out to ffmpeg to transcode the OpenCV-written
file into a broadly-compatible H.264 MP4. OpenCV is used purely for
mechanical frame IO/drawing here -- never for ball detection.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import cv2

from app.config import TRAJECTORY_COLOR_BGR
from app.services.trajectory import build_trajectory_segments
from app.services.video import VideoError

CSV_FIELDNAMES = [
    "id",
    "frame_index",
    "timestamp",
    "ball_visible",
    "x",
    "y",
    "normalized_x",
    "normalized_y",
    "confidence",
    "source_window",
]


class FFmpegNotAvailableError(RuntimeError):
    """Raised when ffmpeg is required for a step but isn't installed."""


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def build_json_export(analysis: Dict[str, Any], detections: List[Dict[str, Any]]) -> bytes:
    payload = {**analysis, "detections": detections}
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def build_csv_export(detections: List[Dict[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    for detection in detections:
        writer.writerow({key: detection.get(key) for key in CSV_FIELDNAMES})
    return buffer.getvalue().encode("utf-8")


def export_trajectory_video(
    video_path: str | Path,
    output_path: str | Path,
    detections: List[Dict[str, Any]],
    width: int,
    height: int,
    fps: float,
    line_thickness: int = 3,
    max_gap_frames: int = 10,
    show_markers: bool = True,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments = build_trajectory_segments(detections, max_gap_frames)
    segment_progress = [0] * len(segments)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoError(f"Could not open source video for export: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = output_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps or 30.0, (width, height))
    if not writer.isOpened():
        cap.release()
        raise VideoError("Could not open OpenCV VideoWriter for export")

    color = TRAJECTORY_COLOR_BGR
    marker_radius = max(3, line_thickness + 2)
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for seg_idx, segment in enumerate(segments):
                while (
                    segment_progress[seg_idx] < len(segment)
                    and segment[segment_progress[seg_idx]]["frame_index"] <= frame_idx
                ):
                    segment_progress[seg_idx] += 1

                visible_points = segment[: segment_progress[seg_idx]]
                for i in range(1, len(visible_points)):
                    p1 = (int(visible_points[i - 1]["x"]), int(visible_points[i - 1]["y"]))
                    p2 = (int(visible_points[i]["x"]), int(visible_points[i]["y"]))
                    cv2.line(frame, p1, p2, color, line_thickness, cv2.LINE_AA)

                if show_markers:
                    for point in visible_points:
                        center = (int(point["x"]), int(point["y"]))
                        cv2.circle(frame, center, marker_radius, color, -1, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            str(tmp_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        tmp_path.unlink(missing_ok=True)
        if result.returncode != 0 or not output_path.exists():
            raise FFmpegNotAvailableError(
                f"ffmpeg failed to transcode exported video: "
                f"{result.stderr.decode(errors='ignore')[:500]}"
            )
    else:
        # No ffmpeg on PATH: fall back to the OpenCV-encoded file directly.
        # It is a valid MP4 (mp4v codec) but less broadly compatible than H.264.
        tmp_path.replace(output_path)

    return output_path
