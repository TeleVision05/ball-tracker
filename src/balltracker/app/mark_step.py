"""Step 2 in marking mode: click the ball on each frame, then calculate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from ..calib_io import load_calibration
from ..pipeline import analyze_from_track
from ..session import add_delivery_analysis, create_session
from ..tracking import track_from_clicks
from ..user_messages import describe_error
from .images import annotated_mark_image
from .logic import (
    MIN_MARKED_FRAMES,
    click_tuples,
    display_to_image,
    marking_progress_text,
    marks_needed_text,
    normalise_clicks,
    prediction_for,
    read_video_frame,
)
from .state import (
    advance_mark_frame,
    exit_mark_mode,
    goto,
    mark_slider_key,
    set_mark_frame,
    video_meta,
)


@dataclass
class _Marking:
    """This run's view of the marking session."""

    path: Path
    fps: float
    start: int
    end: int
    frame: int
    clicks: dict

    @property
    def name(self) -> str:
        return self.path.name


def _current_marking() -> _Marking:
    """Read the marking state, clamp the frame and sync the frame slider."""
    mark_path = Path(st.session_state.mark_video)
    meta = video_meta(mark_path)
    start_f = int(st.session_state.mark_start)
    end_f = int(st.session_state.mark_end)
    frame_f = int(np.clip(st.session_state.mark_frame, start_f, end_f))
    st.session_state.mark_frame = frame_f
    slider_key = mark_slider_key(mark_path.name)
    # Must happen before st.slider(..., key=slider_key) is created.
    if st.session_state.pop("_mark_slider_sync", False) or slider_key not in st.session_state:
        st.session_state[slider_key] = frame_f
    # Store clicks with int keys (Streamlit may revive them as str)
    clicks = normalise_clicks(st.session_state.mark_clicks)
    st.session_state.mark_clicks = clicks
    return _Marking(mark_path, max(1.0, meta["fps"]), start_f, end_f, frame_f, clicks)


def _mark_picture(m: _Marking, pred: tuple[float, float] | None) -> None:
    """The frame with marks; a new click marks the ball and moves on."""
    try:
        frame_bgr, _ = read_video_frame(m.path, m.frame)
    except RuntimeError as exc:
        st.error(describe_error(exc, "loading a frame to mark"))
        return

    display_img, scale = annotated_mark_image(frame_bgr, m.clicks, m.frame, prediction=pred)
    click = streamlit_image_coordinates(
        display_img,
        key=f"mark_click_{m.name}_{m.frame}",
    )

    if click is not None:
        coord = (int(click["x"]), int(click["y"]))
        prev_key = (m.frame, coord)
        if st.session_state.get("_prev_mark_click") != prev_key:
            st.session_state._prev_mark_click = prev_key
            u, v = display_to_image(click, scale)
            m.clicks[m.frame] = [u, v]
            st.session_state.mark_clicks = m.clicks
            advance_mark_frame(m.name)
            st.rerun()


def _frame_slider(m: _Marking) -> None:
    # Frame scrubber (session_state key keeps it in sync with mark_frame)
    new_frame = st.slider(
        "Frame",
        min_value=m.start,
        max_value=m.end,
        key=mark_slider_key(m.name),
    )
    if int(new_frame) != m.frame:
        set_mark_frame(int(new_frame), m.name)
        st.rerun()


def _nav_buttons(m: _Marking, pred: tuple[float, float] | None) -> None:
    nav1, nav2, nav3, nav4 = st.columns(4)
    with nav1:
        if st.button("← Prev", use_container_width=True, disabled=m.frame <= m.start):
            set_mark_frame(m.frame - 1, m.name)
            st.rerun()
    with nav2:
        if st.button("Next →", use_container_width=True, disabled=m.frame >= m.end):
            set_mark_frame(m.frame + 1, m.name)
            st.rerun()
    with nav3:
        can_use_pred = pred is not None and m.frame not in m.clicks
        if st.button(
            "Use prediction",
            use_container_width=True,
            disabled=not can_use_pred,
            help="Accept the predicted ball position for this frame",
        ):
            m.clicks[m.frame] = [float(pred[0]), float(pred[1])]
            st.session_state.mark_clicks = m.clicks
            advance_mark_frame(m.name)
            st.rerun()
    with nav4:
        if st.button(
            "Skip frame",
            use_container_width=True,
            disabled=m.frame >= m.end,
        ):
            advance_mark_frame(m.name)
            st.rerun()


def _calculate(m: _Marking, calib_file: Path, sessions_root: Path) -> None:
    """Fit the trajectory from the marks, save the delivery and show the result."""
    track = track_from_clicks(click_tuples(m.clicks), fps=m.fps, fill_gaps=True)
    if st.session_state.session is None:
        st.session_state.session = create_session(root=sessions_root)
    session = st.session_state.session
    with st.spinner("Fitting trajectory from your clicks…"):
        try:
            calib_obj, _pts = load_calibration(calib_file)
            result = analyze_from_track(
                m.path,
                track,
                calib=calib_obj,
                calib_path=calib_file,
                start_frame=m.start,
                end_frame=m.end,
            )
            delivery_id = add_delivery_analysis(session, m.path, result, write_clip=True)
            st.session_state.session = session
            st.session_state.last_delivery_id = delivery_id
            exit_mark_mode()
            goto("result")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(describe_error(exc, "calculating from the marked clicks"))


def _action_buttons(m: _Marking, calib_file: Path, sessions_root: Path) -> bool:
    """Undo, cancel and calculate; returns whether enough frames are marked."""
    act1, act2, act3 = st.columns(3)
    with act1:
        if st.button("Undo last mark", use_container_width=True, disabled=not m.clicks):
            last = max(m.clicks)
            del m.clicks[last]
            st.session_state.mark_clicks = m.clicks
            set_mark_frame(int(last), m.name)
            st.rerun()
    with act2:
        if st.button("Cancel marking", use_container_width=True):
            exit_mark_mode()
            st.rerun()
    with act3:
        ready = len(m.clicks) >= MIN_MARKED_FRAMES
        if st.button(
            "Calculate speed & metrics",
            type="primary",
            use_container_width=True,
            disabled=not ready,
            help="Need at least 5 marked frames",
        ):
            _calculate(m, calib_file, sessions_root)
    return ready


def render_marking(calib_file: Path, sessions_root: Path) -> None:
    m = _current_marking()
    n_clicked = len(m.clicks)
    st.markdown(marking_progress_text(n_clicked, m.frame, m.start, m.end))

    pred = prediction_for(m.clicks, m.frame, m.fps)
    _mark_picture(m, pred)
    _frame_slider(m)
    _nav_buttons(m, pred)
    if not _action_buttons(m, calib_file, sessions_root):
        st.caption(marks_needed_text(n_clicked))
