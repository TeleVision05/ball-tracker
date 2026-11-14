"""Step 1: set up the camera by clicking stump tips and bases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from ..landmarks import (
    LANDMARK_ORDER,
    LANDMARK_PLAIN_LABELS,
    MIN_LANDMARKS,
    WARN_RMSE_PX,
    calibrate_from_points,
    next_landmark_name,
)
from ..user_messages import describe_error
from .images import annotated_display_image
from .logic import display_to_image, read_video_frame, remove_last_landmark
from .state import goto, video_meta


def _load_picture(video_path: Path, frame_idx: int) -> None:
    try:
        frame, size = read_video_frame(video_path, int(frame_idx))
        st.session_state.calib_frame_bgr = frame
        st.session_state.calib_frame_size = size
        st.session_state.calib_video = str(video_path)
        st.session_state.calib_frame_idx = int(frame_idx)
        st.session_state.calib_points = {}
        st.session_state.calib_rmse = None
        st.session_state._prev_click_coord = None
    except RuntimeError as exc:
        st.error(describe_error(exc, "loading the setup picture"))


def _picture_controls(video_paths: list[Path], folder_path: Path) -> float:
    """Video and frame pickers, the field-of-view setting and Load picture."""
    video_names = [p.name for p in video_paths]
    chosen_name = st.selectbox("Video", video_names, key="setup_video_name")
    video_path = folder_path / chosen_name
    meta = video_meta(video_path)
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
        _load_picture(video_path, frame_idx)
    return fov_deg


def _click_points(points: dict) -> None:
    """Show progress and the picture; a new click places the next point."""
    next_name = next_landmark_name(points)

    st.write(f"**{len(points)} of {len(LANDMARK_ORDER)} points**")
    if next_name:
        st.info(f"Click: **{LANDMARK_PLAIN_LABELS.get(next_name, next_name)}**")
    else:
        st.success("All points placed — save when ready.")

    display_img, scale = annotated_display_image(st.session_state.calib_frame_bgr, points)
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
            u, v = display_to_image(click, scale)
            points[next_name] = np.array([u, v], dtype=np.float64)
            st.session_state.calib_points = points
            st.rerun()


def _save_setup(points: dict, fov_deg: float, calib_file: Path) -> None:
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
            st.warning("Saved, but try clicking more carefully if results look off.")
        goto("analyze")
        st.rerun()
    except (ValueError, RuntimeError) as exc:
        st.error(describe_error(exc, "saving the camera setup"))


def _point_buttons(points: dict, fov_deg: float, calib_file: Path) -> None:
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Undo", disabled=not points, use_container_width=True):
            st.session_state.calib_points = remove_last_landmark(points)
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
            _save_setup(points, fov_deg, calib_file)


def render_setup(video_paths: list[Path], folder_path: Path, calib_file: Path) -> None:
    st.header("Set up your camera")
    st.markdown(
        "Click each stump tip and base on the picture, in the order shown. "
        "**You only do this once for this camera angle.**"
    )

    if not video_paths:
        st.info("Add a video first (use the sidebar) to set up your camera.")
        return
    fov_deg = _picture_controls(video_paths, folder_path)
    if st.session_state.calib_frame_bgr is not None:
        points: dict = st.session_state.calib_points
        _click_points(points)
        _point_buttons(points, fov_deg, calib_file)
