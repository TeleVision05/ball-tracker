"""Step 2: pick and trim a clip, then mark the ball (or run automatic analysis)."""

from __future__ import annotations

from pathlib import Path

import cv2
import streamlit as st

from ..calib_io import load_calibration
from ..pipeline import analyze_delivery
from ..release import mediapipe_available
from ..session import add_delivery_analysis, create_session
from ..user_messages import describe_error
from .logic import (
    clip_length_text,
    read_video_frame,
    suggest_window,
    trim_end_here,
    trim_start_here,
)
from .mark_step import render_marking
from .state import goto, video_meta

DETECTION_METHODS = ["auto", "white", "hsv", "motion"]


def _preview(analyze_path: Path, preview_frame: int, start_frame: int, end_frame: int) -> None:
    """The preview frame, outlined in green when it is inside the clip."""
    try:
        prev_bgr, _ = read_video_frame(analyze_path, preview_frame)
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
        st.error(describe_error(exc, "loading the preview frame"))


def _trim_controls(
    analyze_path: Path, window_key: str, n_frames: int, fps_v: float
) -> tuple[int, int]:
    """Preview slider, start/end buttons and the clip-range slider."""
    start_frame, end_frame = st.session_state[window_key]
    st.caption("Where the clip starts and ends.")
    preview_frame = st.slider(
        "Preview",
        min_value=0,
        max_value=max(0, n_frames - 1),
        value=int(start_frame),
        key=f"preview_{analyze_path.name}",
    )
    _preview(analyze_path, preview_frame, start_frame, end_frame)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Where the clip starts = here"):
            st.session_state[window_key] = trim_start_here(preview_frame, end_frame)
            st.rerun()
    with c2:
        if st.button("Where the clip ends = here"):
            st.session_state[window_key] = trim_end_here(start_frame, preview_frame)
            st.rerun()

    start_frame, end_frame = st.slider(
        "Clip range",
        min_value=0,
        max_value=max(0, n_frames - 1),
        key=window_key,
    )
    st.caption(clip_length_text(start_frame, end_frame, fps_v))
    return start_frame, end_frame


def _start_marking(analyze_path: Path, start_frame: int, end_frame: int) -> None:
    st.session_state.mark_mode = True
    st.session_state.mark_video = str(analyze_path)
    st.session_state.mark_start = int(start_frame)
    st.session_state.mark_end = int(end_frame)
    st.session_state.mark_frame = int(start_frame)
    st.session_state.mark_clicks = {}
    st.session_state._prev_mark_click = None
    st.session_state._mark_slider_sync = True
    st.rerun()


def _run_automatic(
    analyze_path: Path,
    detection_method: str,
    window: tuple[int, int],
    calib_file: Path,
    sessions_root: Path,
) -> None:
    start_frame, end_frame = window
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
            delivery_id = add_delivery_analysis(session, analyze_path, result, write_clip=True)
            st.session_state.session = session
            st.session_state.last_delivery_id = delivery_id
            goto("result")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(describe_error(exc, "analyzing the clip"))


def _automatic_section(
    analyze_path: Path, window: tuple[int, int], calib_file: Path, sessions_root: Path
) -> None:
    with st.expander("Automatic detection (legacy)"):
        if not mediapipe_available():
            st.caption("Hand tip-off unavailable on this computer — using motion tracking instead.")
        detection_method = st.selectbox(
            "Detection method",
            DETECTION_METHODS,
            index=0,
            help="'auto' seeds from the bowler's hand, then tracks with motion. "
            "Use 'white' only for bright training balls.",
        )
        if st.button("Run automatic analyze"):
            if window[1] <= window[0]:
                st.error("The clip end needs to be after the start.")
            else:
                _run_automatic(analyze_path, detection_method, window, calib_file, sessions_root)


def _clip_setup(
    video_paths: list[Path], folder_path: Path, calib_file: Path, sessions_root: Path
) -> None:
    """Pick a video, find or trim the flight, then start marking."""
    analyze_name = st.selectbox(
        "Video",
        [p.name for p in video_paths],
        key="analyze_video_name",
    )
    analyze_path = folder_path / analyze_name

    meta = video_meta(analyze_path)
    n_frames = max(1, meta["n"])
    fps_v = max(1.0, meta["fps"])

    window_key = f"window_{analyze_path.name}"
    if st.button("Find the ball flight"):
        st.session_state[window_key] = suggest_window(analyze_path, n_frames, fps_v)
    if window_key not in st.session_state:
        st.session_state[window_key] = suggest_window(analyze_path, n_frames, fps_v)

    st.markdown(
        "Trim to the delivery, then **click the ball on each frame**. "
        "The app predicts the next position from your clicks and then "
        "calculates speed and bounce metrics."
    )

    with st.expander("Adjust trim", expanded=True):
        start_frame, end_frame = _trim_controls(analyze_path, window_key, n_frames, fps_v)

    if end_frame <= start_frame:
        st.error("The clip end needs to be after the start.")
    elif st.button("Start marking the ball", type="primary"):
        _start_marking(analyze_path, start_frame, end_frame)

    _automatic_section(analyze_path, (start_frame, end_frame), calib_file, sessions_root)


def render_analyze(
    video_paths: list[Path], folder_path: Path, calib_file: Path, sessions_root: Path
) -> None:
    st.header("Analyze a ball")

    if not calib_file.exists():
        st.warning("Set up your camera first.")
        if st.button("Go to camera setup"):
            goto("setup")
            st.rerun()
    elif not video_paths:
        st.info("Add a video first (use the sidebar).")
    elif st.session_state.mark_mode:
        render_marking(calib_file, sessions_root)
    else:
        _clip_setup(video_paths, folder_path, calib_file, sessions_root)
