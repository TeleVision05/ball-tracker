"""Session-state defaults and navigation helpers for the session app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from .logic import next_mark_frame, read_video_meta


def state_defaults(default_real_dir: Path) -> dict:
    """Every session-state key the app reads, with its first-run value."""
    return {
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


def init_state(default_real_dir: Path) -> None:
    for key, value in state_defaults(default_real_dir).items():
        if key not in st.session_state:
            st.session_state[key] = value


def video_meta(video_path: Path) -> dict:
    """Frame count and fps, read once per clip and kept in session state."""
    meta_key = f"meta_{video_path}"
    if meta_key not in st.session_state:
        st.session_state[meta_key] = read_video_meta(video_path)
    return st.session_state[meta_key]


def goto(step: str) -> None:
    st.session_state.ui_step = step


def exit_mark_mode() -> None:
    st.session_state.mark_mode = False
    st.session_state.mark_clicks = {}
    st.session_state._prev_mark_click = None


def mark_slider_key(video_name: str) -> str:
    return f"mark_slider_{video_name}"


def set_mark_frame(frame: int, video_name: str | None = None) -> None:
    """Update the current mark frame; sync the frame slider on the next run."""
    del video_name  # reserved for callers; slider sync is deferred via flag
    st.session_state.mark_frame = int(frame)
    st.session_state._prev_mark_click = None
    # Streamlit forbids mutating a widget key after that widget is drawn in
    # the same run — defer the slider update to the top of the next run.
    st.session_state._mark_slider_sync = True


def advance_mark_frame(video_name: str | None = None) -> None:
    nxt = next_mark_frame(st.session_state.mark_frame, st.session_state.mark_end)
    if nxt is not None:
        set_mark_frame(nxt, video_name)
    else:
        st.session_state._prev_mark_click = None
