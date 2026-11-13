"""OpenCV-backed video helpers: metadata probing, frame extraction, and
frame-index/timestamp conversions.

OpenCV is used ONLY for mechanical frame IO and drawing here -- it never
makes ball-detection decisions. That job belongs entirely to the Ollama
vision model (see services/ollama_client.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_FALLBACK_FPS = 30.0
CONTACT_SHEET_PANEL_WIDTH = 360


class VideoError(Exception):
    """Raised when a video file cannot be opened or read."""


@dataclass(frozen=True)
class EncodedFrame:
    """A JPEG frame prepared for the vision model, with scale-back factors."""

    jpeg_bytes: bytes
    width: int
    height: int
    original_width: int
    original_height: int

    def scale_to_original(self, x: Optional[float], y: Optional[float]) -> tuple:
        if x is None or y is None or not self.width or not self.height:
            return None, None
        return x * (self.original_width / self.width), y * (self.original_height / self.height)


def encode_frame_for_ollama(
    frame_bgr: np.ndarray, max_width: int = 640, quality: int = 70
) -> EncodedFrame:
    """Downscale (if needed) and JPEG-encode a frame for Ollama."""
    original_height, original_width = frame_bgr.shape[:2]
    if original_width > max_width:
        scale = max_width / float(original_width)
        width = max_width
        height = max(1, int(round(original_height * scale)))
        resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    else:
        width, height = original_width, original_height
        resized = frame_bgr
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VideoError("Failed to JPEG-encode frame for Ollama")
    return EncodedFrame(
        jpeg_bytes=buf.tobytes(),
        width=width,
        height=height,
        original_width=original_width,
        original_height=original_height,
    )


def get_video_metadata(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"Could not open video file: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = DEFAULT_FALLBACK_FPS
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            frame_count = _count_frames_by_scan(cap)
    finally:
        cap.release()

    duration = frame_count / fps if fps else 0.0
    return {
        "width": width,
        "height": height,
        "fps": float(fps),
        "frame_count": frame_count,
        "duration": duration,
    }


def _count_frames_by_scan(cap: "cv2.VideoCapture") -> int:
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return count


def extract_frames_bgr(path: str | Path, start: int, end: int) -> List[np.ndarray]:
    """Extract frames [start, end) as BGR numpy arrays."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"Could not open video file: {path}")

    frames: List[np.ndarray] = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(start, end):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def extract_frames_jpeg(path: str | Path, start: int, end: int, quality: int = 85) -> List[bytes]:
    """Extract frames [start, end) from the video as JPEG-encoded bytes."""
    frames: List[bytes] = []
    for frame in extract_frames_bgr(path, start, end):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            frames.append(buf.tobytes())
    return frames


@dataclass(frozen=True)
class ContactSheet:
    """A labeled montage of a temporal window for the vision model."""

    jpeg_bytes: bytes
    panel_width: int
    panel_height: int
    original_width: int
    original_height: int
    frame_count: int

    def scale_to_original(self, x: Optional[float], y: Optional[float]) -> tuple:
        """Map panel-local pixel coords back to original video pixels."""
        if x is None or y is None or not self.panel_width or not self.panel_height:
            return None, None
        scale_x = self.original_width / self.panel_width
        scale_y = self.original_height / self.panel_height
        return x * scale_x, y * scale_y


def build_contact_sheet(
    frames_bgr: List[np.ndarray],
    panel_width: int = CONTACT_SHEET_PANEL_WIDTH,
    quality: int = 70,
) -> ContactSheet:
    """Build a horizontal contact sheet of consecutive frames.

    Vision models (especially thinking VLMs) handle a single annotated collage
    more reliably than many raw images in one chat turn. Coordinates returned
    by the model are expected to be panel-local; use ``scale_to_original``.
    """
    if not frames_bgr:
        raise VideoError("Cannot build contact sheet from zero frames")

    original_height, original_width = frames_bgr[0].shape[:2]
    scale = panel_width / float(original_width)
    panel_height = max(1, int(round(original_height * scale)))

    panels: List[np.ndarray] = []
    for idx, frame in enumerate(frames_bgr):
        resized = cv2.resize(frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
        # Dark label bar so the model can map panel → frame_index.
        bar_h = 22
        labeled = np.zeros((panel_height + bar_h, panel_width, 3), dtype=np.uint8)
        labeled[bar_h:] = resized
        cv2.putText(
            labeled,
            f"frame_index={idx}",
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 208),
            1,
            cv2.LINE_AA,
        )
        panels.append(labeled)

    sheet = np.hstack(panels)
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VideoError("Failed to encode contact sheet JPEG")

    return ContactSheet(
        jpeg_bytes=buf.tobytes(),
        panel_width=panel_width,
        panel_height=panel_height,
        original_width=original_width,
        original_height=original_height,
        frame_count=len(frames_bgr),
    )


def frame_to_timestamp(frame_index: int, fps: float) -> float:
    """Convert a zero-based frame index to a timestamp in seconds."""
    if not fps or fps <= 0:
        return 0.0
    return frame_index / fps


def timestamp_to_frame(timestamp: float, fps: float) -> int:
    """Convert a timestamp in seconds to the nearest zero-based frame index."""
    if not fps or fps <= 0:
        return 0
    return round(timestamp * fps)


def compute_windows(
    frame_count: int, window_size: int, window_overlap: int
) -> List[Tuple[int, int]]:
    """Compute sliding-window (start, end) frame ranges covering the video.

    Windows overlap by `window_overlap` frames. The final window is adjusted
    to end exactly at `frame_count` so no trailing frames are ever skipped.
    """
    if frame_count <= 0:
        return []

    window_size = max(1, window_size)
    overlap = min(max(0, window_overlap), window_size - 1)
    stride = max(1, window_size - overlap)

    if frame_count <= window_size:
        return [(0, frame_count)]

    windows: List[Tuple[int, int]] = []
    start = 0
    while start < frame_count:
        end = min(start + window_size, frame_count)
        windows.append((start, end))
        if end == frame_count:
            break
        start += stride
    return windows
