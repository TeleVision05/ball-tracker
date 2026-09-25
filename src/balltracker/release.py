"""MediaPipe-based release-point seeding for the hybrid tracker.

MediaPipe is a soft dependency: any import/runtime failure (not installed,
no pose found, unsupported build, etc.) degrades to ``None`` rather than
raising, so the rest of the pipeline can fall back to unseeded detection.

Supports both the classic ``mp.solutions.pose`` API and the newer Tasks
``PoseLandmarker`` API (required on newer mediapipe wheels that omit solutions).
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import load_config

# Pose landmark indices (consistent across solutions + tasks PoseLandmark enums).
_LEFT_WRIST = 15
_RIGHT_WRIST = 16

_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


@dataclass
class ReleaseSeed:
    """Estimated bowling-arm release point/direction from pose landmarks."""

    frame_idx: int
    u: float
    v: float
    direction_uv: tuple[float, float]
    confidence: float


def mediapipe_available() -> bool:
    """Whether a usable MediaPipe Pose backend is importable."""
    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            return True
        from mediapipe.tasks.python import vision  # noqa: F401

        return hasattr(vision, "PoseLandmarker")
    except Exception:
        return False


def _pose_model_path(config: dict) -> Path:
    models_dir = Path(config.get("paths", {}).get("models", "models"))
    if not models_dir.is_absolute():
        # Resolve relative to package repo root (…/BallTracker/models).
        models_dir = Path(__file__).resolve().parents[2] / models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir / "pose_landmarker_lite.task"


def _ensure_pose_model(config: dict) -> Path | None:
    """Download the lite Pose Landmarker bundle on first use."""
    path = _pose_model_path(config)
    if path.exists() and path.stat().st_size > 1000:
        return path
    try:
        urllib.request.urlretrieve(_POSE_MODEL_URL, path)
        if path.exists() and path.stat().st_size > 1000:
            return path
    except Exception:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
    return None


def _wrist_track_solutions(
    video_path: str | Path,
    start_frame: int,
    window_end: int,
    stride: int,
    config: dict,
) -> tuple[list[tuple[int, float, float, float]], list[tuple[int, float, float, float]], float]:
    """Return (left_wrist, right_wrist, fps) via mp.solutions.pose."""
    import mediapipe as mp

    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1:
        fps = float(config.get("camera", {}).get("fps", 120))

    left: list[tuple[int, float, float, float]] = []
    right: list[tuple[int, float, float, float]] = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as pose:
        frame_idx = start_frame
        while frame_idx <= window_end:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_idx - start_frame) % max(1, stride) == 0:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                landmarks = getattr(result, "pose_landmarks", None)
                if landmarks is not None:
                    lm = landmarks.landmark
                    lw = lm[_LEFT_WRIST]
                    rw = lm[_RIGHT_WRIST]
                    left.append(
                        (frame_idx, lw.x * w, lw.y * h, float(getattr(lw, "visibility", 1.0)))
                    )
                    right.append(
                        (frame_idx, rw.x * w, rw.y * h, float(getattr(rw, "visibility", 1.0)))
                    )
            frame_idx += 1
    cap.release()
    return left, right, fps


def _wrist_track_tasks(
    video_path: str | Path,
    start_frame: int,
    window_end: int,
    stride: int,
    config: dict,
) -> tuple[list[tuple[int, float, float, float]], list[tuple[int, float, float, float]], float]:
    """Return (left_wrist, right_wrist, fps) via MediaPipe Tasks PoseLandmarker."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options as base_options_module

    model_path = _ensure_pose_model(config)
    if model_path is None:
        return [], [], 0.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1:
        fps = float(config.get("camera", {}).get("fps", 120))

    left: list[tuple[int, float, float, float]] = []
    right: list[tuple[int, float, float, float]] = []

    options = vision.PoseLandmarkerOptions(
        base_options=base_options_module.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = start_frame
        while frame_idx <= window_end:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_idx - start_frame) % max(1, stride) == 0:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                timestamp_ms = int(round(frame_idx * 1000.0 / max(fps, 1e-6)))
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if result.pose_landmarks:
                    # Prefer the pose whose wrists sit lower in the frame (bowler nearer camera).
                    best = max(
                        result.pose_landmarks,
                        key=lambda lm: 0.5 * (lm[_LEFT_WRIST].y + lm[_RIGHT_WRIST].y),
                    )
                    lw = best[_LEFT_WRIST]
                    rw = best[_RIGHT_WRIST]
                    left.append(
                        (
                            frame_idx,
                            lw.x * w,
                            lw.y * h,
                            float(getattr(lw, "visibility", 1.0) or 1.0),
                        )
                    )
                    right.append(
                        (
                            frame_idx,
                            rw.x * w,
                            rw.y * h,
                            float(getattr(rw, "visibility", 1.0) or 1.0),
                        )
                    )
            frame_idx += 1
    cap.release()
    return left, right, fps


def _wrist_track(
    video_path: str | Path,
    start_frame: int,
    window_end: int,
    stride: int,
    config: dict,
) -> tuple[list[tuple[int, float, float, float]], list[tuple[int, float, float, float]], float]:
    """Dispatch to solutions or tasks Pose backend."""
    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            return _wrist_track_solutions(video_path, start_frame, window_end, stride, config)
    except Exception:
        pass
    return _wrist_track_tasks(video_path, start_frame, window_end, stride, config)


def _seed_from_wrist_track(
    positions: list[tuple[int, float, float, float]],
    fps: float,
    frame_diag_px: float,
) -> ReleaseSeed | None:
    """Peak-speed release seed for one wrist's tracked positions."""
    if len(positions) < 3:
        return None

    speeds: list[float] = []
    for (f0, u0, v0, _vis0), (f1, u1, v1, _vis1) in zip(positions, positions[1:]):
        dt = max(1, f1 - f0) / max(fps, 1e-6)
        dist = float(np.hypot(u1 - u0, v1 - v0))
        speeds.append(dist / dt)

    if not speeds:
        return None
    peak_i = int(np.argmax(speeds)) + 1  # index into `positions` at the end of the fastest segment
    peak_speed = speeds[peak_i - 1]
    if peak_speed <= 1e-6:
        return None

    release_idx, release_u, release_v, visibility = positions[peak_i]

    lo = max(0, peak_i - 3)
    du = positions[peak_i][1] - positions[lo][1]
    dv = positions[peak_i][2] - positions[lo][2]
    norm = float(np.hypot(du, dv))
    direction = (du / norm, dv / norm) if norm > 1e-6 else (0.0, -1.0)

    # Reference "fast" speed: covering ~60% of the frame diagonal per second is a
    # brisk bowling-arm swing; used only to squash speed into a 0-1 confidence term.
    speed_ref = max(1.0, 0.6 * frame_diag_px)
    norm_speed = float(np.clip(peak_speed / speed_ref, 0.0, 1.0))
    confidence = float(np.clip(visibility * norm_speed, 0.0, 1.0))

    return ReleaseSeed(
        frame_idx=int(release_idx),
        u=float(release_u),
        v=float(release_v),
        direction_uv=direction,
        confidence=confidence,
    )


def estimate_release_seed(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    config: dict | None = None,
) -> ReleaseSeed | None:
    """Estimate a ball-release seed (position + direction) from pose landmarks.

    Looks at wrist motion in the first ~1.5s (or 90 frames) of the analysis
    window; the release frame is approximated by the peak wrist speed across
    both wrists (the faster one is treated as the bowling arm).
    """
    if not mediapipe_available():
        return None
    try:
        if config is None:
            config = load_config()
        det_cfg = config.get("detection", {})
        stride = int(det_cfg.get("pose_stride", 2))
        max_window_frames = int(det_cfg.get("pose_window_frames", 90))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1:
            fps = float(config.get("camera", {}).get("fps", 120))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
        cap.release()

        if end_frame is None or end_frame < 0:
            end_frame = max(0, total - 1)
        start_frame = int(max(0, start_frame))
        window_end = min(
            end_frame, start_frame + max_window_frames, start_frame + int(round(1.5 * fps))
        )
        window_end = max(window_end, start_frame)

        left, right, fps = _wrist_track(video_path, start_frame, window_end, stride, config)
        frame_diag_px = float(np.hypot(w, h)) if w > 0 and h > 0 else 1000.0

        candidates = [
            seed
            for seed in (
                _seed_from_wrist_track(left, fps, frame_diag_px),
                _seed_from_wrist_track(right, fps, frame_diag_px),
            )
            if seed is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.confidence)
    except Exception:
        return None
