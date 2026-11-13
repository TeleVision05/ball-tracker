#!/usr/bin/env python3
"""Streamlit session app: 3-step senior-friendly bowl analysis flow."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Prefer the live source tree over any stale installed egg.
for _name in list(sys.modules):
    if _name == "balltracker" or _name.startswith("balltracker."):
        del sys.modules[_name]

from balltracker.calib_io import load_calibration
from balltracker.config import load_config
from balltracker.flight_window import suggest_flight_window
from balltracker.landmarks import (
    LANDMARK_ORDER,
    LANDMARK_PLAIN_LABELS,
    MIN_LANDMARKS,
    WARN_RMSE_PX,
    calibrate_from_points,
    next_landmark_name,
)
from balltracker.pipeline import analyze_delivery, analyze_from_track
from balltracker.plain_language import summarize_delivery
from balltracker.release import mediapipe_available
from balltracker.session import (
    add_delivery_analysis,
    create_session,
    list_deliveries,
    load_session,
)
from balltracker.tracking import predict_ball_uv, track_from_clicks
from balltracker.viz import draw_pitch_map, ensure_web_playable

st.set_page_config(page_title="BallTracker", layout="wide")
st.title("BallTracker — bowl analysis")
st.caption("Film behind the stumps, then see how fast and where it bounced.")

config = load_config()
default_real_dir = ROOT / config.get("paths", {}).get("raw_video", "data/raw/real")
sessions_root = Path(config.get("paths", {}).get("sessions", ROOT / "sessions"))
sessions_root.mkdir(parents=True, exist_ok=True)
default_real_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_defaults = {
    "ui_step": "setup",
    "session": None,
    "calib_points": {},
    "calib_frame_bgr": None,
    "calib_frame_size": None,  # (w, h) original
    "calib_display_scale": 1.0,
    "calib_video": None,
    "calib_frame_idx": 0,
    "calib_path": str(default_real_dir / "session.calib.json"),
    "calib_rmse": None,
    "last_delivery_id": None,
    "flight_window": None,
    "_routed": False,
    # Manual ball-marking workflow
    "mark_mode": False,
    "mark_clicks": {},  # frame_idx -> [u, v]
    "mark_frame": 0,
    "mark_video": None,
    "mark_start": 0,
    "mark_end": 0,
    "_prev_mark_click": None,
}
for key, value in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def _list_videos(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".mp4", ".mov"})


def _read_video_frame(video_path: Path, frame_idx: int) -> tuple[np.ndarray, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = int(np.clip(frame_idx, 0, max(0, n - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {frame_idx}")
    return frame, (w, h)


def _video_meta(video_path: Path) -> dict:
    meta_key = f"meta_{video_path}"
    if meta_key not in st.session_state:
        cap = cv2.VideoCapture(str(video_path))
        st.session_state[meta_key] = {
            "n": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        }
        cap.release()
    return st.session_state[meta_key]


def _annotated_display_image(
    frame_bgr: np.ndarray,
    points: dict,
    max_width: int = 960,
) -> tuple[Image.Image, float]:
    """Return RGB PIL image with markers and the display scale (display/original)."""
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_width / float(w))
    if scale < 1.0:
        disp = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        disp = frame_bgr.copy()
        scale = 1.0

    for i, (name, uv) in enumerate(points.items()):
        u = int(float(uv[0]) * scale)
        v = int(float(uv[1]) * scale)
        cv2.drawMarker(disp, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(
            disp,
            str(i + 1),
            (u + 8, max(16, v - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    next_name = next_landmark_name(points)
    if next_name:
        label = f"Click: {LANDMARK_PLAIN_LABELS.get(next_name, next_name)}"
    else:
        label = "All points placed — save when ready"
    cv2.putText(
        disp,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), scale


def _annotated_mark_image(
    frame_bgr: np.ndarray,
    clicks: dict,
    current_frame: int,
    prediction: tuple[float, float] | None = None,
    trail_frames: int = 12,
    max_width: int = 960,
) -> tuple[Image.Image, float]:
    """Frame with prior click trail, current mark, and optional prediction guide."""
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_width / float(w))
    if scale < 1.0:
        disp = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        disp = frame_bgr.copy()
        scale = 1.0

    # Trail of nearby clicks
    nearby = sorted(f for f in clicks if abs(int(f) - current_frame) <= trail_frames)
    pts = []
    for f in nearby:
        uv = clicks[f]
        pts.append((int(float(uv[0]) * scale), int(float(uv[1]) * scale)))
    if len(pts) >= 2:
        cv2.polylines(disp, [np.array(pts, dtype=np.int32)], False, (0, 200, 255), 2, cv2.LINE_AA)
    for f in nearby:
        uv = clicks[f]
        u = int(float(uv[0]) * scale)
        v = int(float(uv[1]) * scale)
        color = (0, 255, 0) if int(f) == current_frame else (0, 180, 255)
        cv2.drawMarker(disp, (u, v), color, cv2.MARKER_CROSS, 14, 2)

    if prediction is not None and current_frame not in clicks:
        pu = int(float(prediction[0]) * scale)
        pv = int(float(prediction[1]) * scale)
        cv2.circle(disp, (pu, pv), 18, (255, 80, 80), 2, cv2.LINE_AA)
        cv2.drawMarker(disp, (pu, pv), (255, 80, 80), cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(
            disp,
            "predicted",
            (pu + 12, max(16, pv - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 80, 80),
            1,
            cv2.LINE_AA,
        )

    label = f"Frame {current_frame} — click the ball"
    cv2.putText(
        disp,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), scale


def _goto(step: str) -> None:
    st.session_state.ui_step = step


def _exit_mark_mode() -> None:
    st.session_state.mark_mode = False
    st.session_state.mark_clicks = {}
    st.session_state._prev_mark_click = None


def _mark_slider_key(video_name: str) -> str:
    return f"mark_slider_{video_name}"


def _set_mark_frame(frame: int, video_name: str | None = None) -> None:
    """Update the current mark frame; sync the frame slider on the next run."""
    del video_name  # reserved for callers; slider sync is deferred via flag
    st.session_state.mark_frame = int(frame)
    st.session_state._prev_mark_click = None
    # Streamlit forbids mutating a widget key after that widget is drawn in
    # the same run — defer the slider update to the top of the next run.
    st.session_state._mark_slider_sync = True


def _advance_mark_frame(video_name: str | None = None) -> None:
    nxt = int(st.session_state.mark_frame) + 1
    end = int(st.session_state.mark_end)
    if nxt <= end:
        _set_mark_frame(nxt, video_name)
    else:
        st.session_state._prev_mark_click = None


# ---------------------------------------------------------------------------
# Sidebar: minimal — video upload/folder + session
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Videos")
    uploaded = st.file_uploader("Upload a video (.mp4 / .mov)", type=["mp4", "mov"])

    with st.expander("Video folder", expanded=False):
        folder = st.text_input("Folder path", value=str(default_real_dir))
    folder_path = Path(folder).expanduser()
    folder_path.mkdir(parents=True, exist_ok=True)

    if uploaded is not None:
        dest = folder_path / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        st.success(f"Saved {dest.name}")

    video_paths = _list_videos(folder_path)
    st.caption(f"{len(video_paths)} video(s) available")

    st.session_state.calib_path = str(folder_path / "session.calib.json")
    st.divider()
    calib_file = Path(st.session_state.calib_path).expanduser()
    if calib_file.exists():
        st.success("Camera setup saved")
    else:
        st.info("Camera setup needed")

    st.divider()
    st.subheader("Session")
    if st.session_state.session is not None:
        st.caption(f"Active: `{st.session_state.session.id}`")
    existing = sorted(sessions_root.glob("*/session.json"))
    existing_labels = ["(new session)"] + [p.parent.name for p in existing]
    pick = st.selectbox("Load an existing session", existing_labels, index=0)
    if pick != "(new session)" and st.button("Load"):
        st.session_state.session = load_session(sessions_root / pick)
        st.success(f"Loaded {pick}")


# ---------------------------------------------------------------------------
# Step routing
# ---------------------------------------------------------------------------
if not st.session_state._routed:
    st.session_state._routed = True
    if not calib_file.exists():
        st.session_state.ui_step = "setup"
    elif st.session_state.last_delivery_id is None:
        st.session_state.ui_step = "analyze"

step_labels = {
    "setup": "1. Set up camera",
    "analyze": "2. Analyze a ball",
    "result": "3. See result",
}
step_order = ["setup", "analyze", "result"]
chosen_label = st.radio(
    "Steps",
    [step_labels[s] for s in step_order],
    index=step_order.index(st.session_state.ui_step),
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state.ui_step = step_order[[step_labels[s] for s in step_order].index(chosen_label)]
st.divider()

# ===========================================================================
# Step 1 — Set up camera
# ===========================================================================
if st.session_state.ui_step == "setup":
    st.header("Set up your camera")
    st.markdown(
        "Click each stump tip and base on the picture, in the order shown. "
        "**You only do this once for this camera angle.**"
    )

    if not video_paths:
        st.info("Add a video first (use the sidebar) to set up your camera.")
    else:
        video_names = [p.name for p in video_paths]
        chosen_name = st.selectbox("Video", video_names, key="setup_video_name")
        video_path = folder_path / chosen_name
        meta = _video_meta(video_path)
        n_setup = max(1, meta["n"])
        frame_idx = st.slider(
            "Pick a clear frame (stumps visible)",
            min_value=0,
            max_value=max(0, n_setup - 1),
            value=0,
            key="setup_frame_idx",
        )
        with st.expander("Advanced"):
            fov_deg = st.number_input(
                "Camera field of view (deg)", min_value=20.0, max_value=120.0, value=55.0
            )

        if st.button("Load picture", type="primary"):
            try:
                frame, size = _read_video_frame(video_path, int(frame_idx))
                st.session_state.calib_frame_bgr = frame
                st.session_state.calib_frame_size = size
                st.session_state.calib_video = str(video_path)
                st.session_state.calib_frame_idx = int(frame_idx)
                st.session_state.calib_points = {}
                st.session_state.calib_rmse = None
                st.session_state._prev_click_coord = None
            except RuntimeError as exc:
                st.error(str(exc))

        if st.session_state.calib_frame_bgr is not None:
            points: dict = st.session_state.calib_points
            next_name = next_landmark_name(points)

            st.write(f"**{len(points)} of {len(LANDMARK_ORDER)} points**")
            if next_name:
                st.info(f"Click: **{LANDMARK_PLAIN_LABELS.get(next_name, next_name)}**")
            else:
                st.success("All points placed — save when ready.")

            display_img, scale = _annotated_display_image(st.session_state.calib_frame_bgr, points)
            st.session_state.calib_display_scale = scale

            click = streamlit_image_coordinates(
                display_img,
                key="calib_click_image",
            )

            # Component echoes the last click across reruns — only accept a new pixel.
            if click is not None and next_name is not None:
                coord = (int(click["x"]), int(click["y"]))
                if st.session_state.get("_prev_click_coord") != coord:
                    st.session_state._prev_click_coord = coord
                    u = float(click["x"]) / scale
                    v = float(click["y"]) / scale
                    points[next_name] = np.array([u, v], dtype=np.float64)
                    st.session_state.calib_points = points
                    st.rerun()

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("Undo", disabled=not points, use_container_width=True):
                    for name in reversed(LANDMARK_ORDER):
                        if name in points:
                            del points[name]
                            break
                    st.session_state.calib_points = points
                    st.session_state._prev_click_coord = None
                    st.rerun()
            with b2:
                if st.button("Reset", disabled=not points, use_container_width=True):
                    st.session_state.calib_points = {}
                    st.session_state.calib_rmse = None
                    st.session_state._prev_click_coord = None
                    st.rerun()
            with b3:
                save_disabled = len(points) < MIN_LANDMARKS
                if st.button(
                    "Save camera setup",
                    type="primary",
                    disabled=save_disabled,
                    use_container_width=True,
                ):
                    try:
                        _calib, rmse = calibrate_from_points(
                            {k: np.asarray(v) for k, v in points.items()},
                            image_size=st.session_state.calib_frame_size,
                            fov_deg=float(fov_deg),
                            out_path=calib_file,
                        )
                        st.session_state.calib_rmse = rmse
                        if rmse <= WARN_RMSE_PX:
                            st.success("Looks good — camera setup saved.")
                        else:
                            st.warning(
                                "Saved, but try clicking more carefully if results look off."
                            )
                        _goto("analyze")
                        st.rerun()
                    except ValueError:
                        st.error("Try again — click closer to the stump tips and bases.")

# ===========================================================================
# Step 2 — Analyze a ball (manual click tracking)
# ===========================================================================
elif st.session_state.ui_step == "analyze":
    st.header("Analyze a ball")

    if not calib_file.exists():
        st.warning("Set up your camera first.")
        if st.button("Go to camera setup"):
            _goto("setup")
            st.rerun()
    elif not video_paths:
        st.info("Add a video first (use the sidebar).")
    elif st.session_state.mark_mode:
        # ----- Frame-by-frame ball marking -----
        mark_path = Path(st.session_state.mark_video)
        meta = _video_meta(mark_path)
        fps_v = max(1.0, meta["fps"])
        start_f = int(st.session_state.mark_start)
        end_f = int(st.session_state.mark_end)
        frame_f = int(np.clip(st.session_state.mark_frame, start_f, end_f))
        st.session_state.mark_frame = frame_f
        video_name = mark_path.name
        slider_key = _mark_slider_key(video_name)
        # Must happen before st.slider(..., key=slider_key) is created.
        if st.session_state.pop("_mark_slider_sync", False) or slider_key not in st.session_state:
            st.session_state[slider_key] = frame_f
        clicks: dict = st.session_state.mark_clicks

        # Store clicks with int keys (Streamlit may revive them as str)
        clicks = {int(k): v for k, v in clicks.items()}
        st.session_state.mark_clicks = clicks

        n_clicked = len(clicks)
        total_in_window = end_f - start_f + 1
        st.markdown(
            f"Click the **ball** on each frame. A red ring shows where the track "
            f"predicts it should be from your previous clicks. "
            f"**{n_clicked}** marked · frame **{frame_f}** of **{start_f}–{end_f}** "
            f"({total_in_window} frames)."
        )

        try:
            frame_bgr, _ = _read_video_frame(mark_path, frame_f)
        except RuntimeError as exc:
            st.error(str(exc))
            frame_bgr = None

        pred = None
        if n_clicked >= 1 and frame_f not in clicks:
            click_tuples = {f: (float(uv[0]), float(uv[1])) for f, uv in clicks.items()}
            pred = predict_ball_uv(click_tuples, frame_f, fps_v)

        if frame_bgr is not None:
            display_img, scale = _annotated_mark_image(frame_bgr, clicks, frame_f, prediction=pred)
            click = streamlit_image_coordinates(
                display_img,
                key=f"mark_click_{video_name}_{frame_f}",
            )

            if click is not None:
                coord = (int(click["x"]), int(click["y"]))
                prev_key = (frame_f, coord)
                if st.session_state.get("_prev_mark_click") != prev_key:
                    st.session_state._prev_mark_click = prev_key
                    u = float(click["x"]) / scale
                    v = float(click["y"]) / scale
                    clicks[frame_f] = [u, v]
                    st.session_state.mark_clicks = clicks
                    _advance_mark_frame(video_name)
                    st.rerun()

        # Frame scrubber (session_state key keeps it in sync with mark_frame)
        new_frame = st.slider(
            "Frame",
            min_value=start_f,
            max_value=end_f,
            key=slider_key,
        )
        if int(new_frame) != frame_f:
            _set_mark_frame(int(new_frame), video_name)
            st.rerun()

        nav1, nav2, nav3, nav4 = st.columns(4)
        with nav1:
            if st.button("← Prev", use_container_width=True, disabled=frame_f <= start_f):
                _set_mark_frame(frame_f - 1, video_name)
                st.rerun()
        with nav2:
            if st.button("Next →", use_container_width=True, disabled=frame_f >= end_f):
                _set_mark_frame(frame_f + 1, video_name)
                st.rerun()
        with nav3:
            can_use_pred = pred is not None and frame_f not in clicks
            if st.button(
                "Use prediction",
                use_container_width=True,
                disabled=not can_use_pred,
                help="Accept the predicted ball position for this frame",
            ):
                clicks[frame_f] = [float(pred[0]), float(pred[1])]
                st.session_state.mark_clicks = clicks
                _advance_mark_frame(video_name)
                st.rerun()
        with nav4:
            if st.button(
                "Skip frame",
                use_container_width=True,
                disabled=frame_f >= end_f,
            ):
                _advance_mark_frame(video_name)
                st.rerun()

        act1, act2, act3 = st.columns(3)
        with act1:
            if st.button("Undo last mark", use_container_width=True, disabled=not clicks):
                last = max(clicks)
                del clicks[last]
                st.session_state.mark_clicks = clicks
                _set_mark_frame(int(last), video_name)
                st.rerun()
        with act2:
            if st.button("Cancel marking", use_container_width=True):
                _exit_mark_mode()
                st.rerun()
        with act3:
            ready = n_clicked >= 5
            if st.button(
                "Calculate speed & metrics",
                type="primary",
                use_container_width=True,
                disabled=not ready,
                help="Need at least 5 marked frames",
            ):
                click_tuples = {int(f): (float(uv[0]), float(uv[1])) for f, uv in clicks.items()}
                track = track_from_clicks(click_tuples, fps=fps_v, fill_gaps=True)
                if st.session_state.session is None:
                    st.session_state.session = create_session(root=sessions_root)
                session = st.session_state.session
                with st.spinner("Fitting trajectory from your clicks…"):
                    try:
                        calib_obj, _pts = load_calibration(calib_file)
                        result = analyze_from_track(
                            mark_path,
                            track,
                            calib=calib_obj,
                            calib_path=calib_file,
                            start_frame=start_f,
                            end_frame=end_f,
                        )
                        delivery_id = add_delivery_analysis(
                            session, mark_path, result, write_clip=True
                        )
                        st.session_state.session = session
                        st.session_state.last_delivery_id = delivery_id
                        _exit_mark_mode()
                        _goto("result")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Couldn't calculate from those clicks: {exc}")

        if not ready:
            st.caption(f"Mark at least **5** frames ({5 - n_clicked} more) to calculate.")

    else:
        # ----- Configure clip, then start marking -----
        analyze_name = st.selectbox(
            "Video",
            [p.name for p in video_paths],
            key="analyze_video_name",
        )
        analyze_path = folder_path / analyze_name

        meta = _video_meta(analyze_path)
        n_frames = max(1, meta["n"])
        fps_v = max(1.0, meta["fps"])

        window_key = f"window_{analyze_path.name}"
        if st.button("Find the ball flight"):
            try:
                st.session_state[window_key] = suggest_flight_window(analyze_path)
            except Exception:
                st.session_state[window_key] = (0, min(n_frames - 1, int(fps_v * 3)))
        if window_key not in st.session_state:
            try:
                st.session_state[window_key] = suggest_flight_window(analyze_path)
            except Exception:
                st.session_state[window_key] = (0, min(n_frames - 1, int(fps_v * 3)))

        start_frame, end_frame = st.session_state[window_key]

        st.markdown(
            "Trim to the delivery, then **click the ball on each frame**. "
            "The app predicts the next position from your clicks and then "
            "calculates speed and bounce metrics."
        )

        with st.expander("Adjust trim", expanded=True):
            st.caption("Where the clip starts and ends.")
            preview_frame = st.slider(
                "Preview",
                min_value=0,
                max_value=max(0, n_frames - 1),
                value=int(start_frame),
                key=f"preview_{analyze_path.name}",
            )
            try:
                prev_bgr, _ = _read_video_frame(analyze_path, preview_frame)
                prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
                if start_frame <= preview_frame <= end_frame:
                    cv2.rectangle(
                        prev_rgb,
                        (0, 0),
                        (prev_rgb.shape[1] - 1, prev_rgb.shape[0] - 1),
                        (0, 200, 0),
                        4,
                    )
                st.image(prev_rgb, use_container_width=True)
            except RuntimeError as exc:
                st.error(str(exc))

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Where the clip starts = here"):
                    st.session_state[window_key] = (
                        int(preview_frame),
                        int(max(preview_frame, end_frame)),
                    )
                    st.rerun()
            with c2:
                if st.button("Where the clip ends = here"):
                    st.session_state[window_key] = (
                        int(min(start_frame, preview_frame)),
                        int(preview_frame),
                    )
                    st.rerun()

            start_frame, end_frame = st.slider(
                "Clip range",
                min_value=0,
                max_value=max(0, n_frames - 1),
                key=window_key,
            )
            dur = (end_frame - start_frame + 1) / fps_v
            st.caption(f"{dur:.2f} seconds · {end_frame - start_frame + 1} frames")

        if end_frame <= start_frame:
            st.error("The clip end needs to be after the start.")
        else:
            if st.button("Start marking the ball", type="primary"):
                st.session_state.mark_mode = True
                st.session_state.mark_video = str(analyze_path)
                st.session_state.mark_start = int(start_frame)
                st.session_state.mark_end = int(end_frame)
                st.session_state.mark_frame = int(start_frame)
                st.session_state.mark_clicks = {}
                st.session_state._prev_mark_click = None
                st.session_state._mark_slider_sync = True
                st.rerun()

        with st.expander("Automatic detection (legacy)"):
            if not mediapipe_available():
                st.caption(
                    "Hand tip-off unavailable on this computer — using motion tracking instead."
                )
            detection_method = st.selectbox(
                "Detection method",
                ["auto", "white", "hsv", "motion"],
                index=0,
                help="'auto' seeds from the bowler's hand, then tracks with motion. "
                "Use 'white' only for bright training balls.",
            )
            if st.button("Run automatic analyze"):
                if end_frame <= start_frame:
                    st.error("The clip end needs to be after the start.")
                else:
                    if st.session_state.session is None:
                        st.session_state.session = create_session(root=sessions_root)
                    session = st.session_state.session
                    with st.spinner("Looking for the ball from the bowler's hand…"):
                        try:
                            calib_obj, _pts = load_calibration(calib_file)
                            result = analyze_delivery(
                                analyze_path,
                                calib=calib_obj,
                                calib_path=calib_file,
                                detection_method=detection_method,
                                start_frame=int(start_frame),
                                end_frame=int(end_frame),
                            )
                            delivery_id = add_delivery_analysis(
                                session, analyze_path, result, write_clip=True
                            )
                            st.session_state.session = session
                            st.session_state.last_delivery_id = delivery_id
                            _goto("result")
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Couldn't analyze that clip: {exc}")

# ===========================================================================
# Step 3 — See result
# ===========================================================================
else:
    st.header("Result")
    session = st.session_state.session
    if session is None:
        st.info("Analyze a ball first.")
        if st.button("Go to step 2"):
            _goto("analyze")
            st.rerun()
    else:
        deliveries = list_deliveries(session)
        if not deliveries:
            st.info("Analyze a ball first.")
            if st.button("Go to step 2"):
                _goto("analyze")
                st.rerun()
        else:
            delivery_ids = [d["delivery_id"] for d in deliveries]
            default_id = st.session_state.last_delivery_id or delivery_ids[-1]

            if len(delivery_ids) > 1:
                selected = st.selectbox(
                    "Previous deliveries",
                    delivery_ids,
                    index=delivery_ids.index(default_id)
                    if default_id in delivery_ids
                    else len(delivery_ids) - 1,
                )
            else:
                selected = delivery_ids[0]

            sel_rec = next(d for d in deliveries if d["delivery_id"] == selected)
            metrics = sel_rec.get("metrics", {})
            summary = summarize_delivery(metrics)

            m1, m2 = st.columns(2)
            with m1:
                speed = summary.get("speed_kmh")
                st.metric("Speed", f"{speed:.1f} km/h" if speed is not None else "—")
            with m2:
                st.markdown("**Where it bounced**")
                st.markdown(f"### {summary.get('bounce_text', '—')}")

            col_map, col_media = st.columns(2)
            with col_map:
                st.markdown("**Pitch map**")
                fig = draw_pitch_map(metrics)
                st.pyplot(fig)
                plt.close(fig)

            with col_media:
                st.markdown("**Video replay**")
                overlay = sel_rec.get("overlay_path")
                overlay_path = Path(overlay) if overlay else None
                played = False
                if (
                    overlay_path is not None
                    and overlay_path.exists()
                    and overlay_path.stat().st_size > 0
                ):
                    try:
                        overlay_path = ensure_web_playable(overlay_path)
                        st.video(overlay_path.read_bytes())
                        played = True
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Could not play the video: {exc}")
                if not played:
                    src = sel_rec.get("source_video_path")
                    src_path = Path(src) if src else None
                    if src_path is not None and src_path.exists():
                        st.video(src_path.read_bytes())
                    else:
                        st.error("No video found on disk.")

            with st.expander("More details"):
                st.markdown(
                    f"Swing: {metrics.get('beta_s_deg', '—')}°  \n"
                    f"Turn: {metrics.get('turn_deg', '—')}°  \n"
                    f"Line: {metrics.get('pitch_line_m', '—')} m  \n"
                    f"Length: {metrics.get('pitch_length_m', '—')} m"
                )

            if st.button("Analyze another ball", type="primary"):
                st.session_state.last_delivery_id = None
                _goto("analyze")
                st.rerun()
