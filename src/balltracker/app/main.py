"""The session app's page: header, paths, routing, step picker and the current step."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..config import load_config
from .analyze_step import render_analyze
from .logic import STEP_LABELS, STEP_ORDER, app_paths, initial_step, step_from_label
from .result_step import render_result
from .setup_step import render_setup
from .sidebar import render_sidebar
from .state import init_state


def _route_first_run(calib_file: Path) -> None:
    """On the first run, open setup without a calibration, else analyze."""
    if not st.session_state._routed:
        st.session_state._routed = True
        st.session_state.ui_step = initial_step(
            st.session_state.ui_step, calib_file.exists(), st.session_state.last_delivery_id
        )


def _step_picker() -> None:
    labels = [STEP_LABELS[s] for s in STEP_ORDER]
    chosen_label = st.radio(
        "Steps",
        labels,
        index=STEP_ORDER.index(st.session_state.ui_step),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.ui_step = step_from_label(chosen_label)
    st.divider()


def main(root: Path) -> None:
    """Draw one run of the app. ``root`` is the project root (default video folder)."""
    st.set_page_config(page_title="BallTracker", layout="wide")
    st.title("BallTracker — bowl analysis")
    st.caption("Film behind the stumps, then see how fast and where it bounced.")

    default_real_dir, sessions_root = app_paths(load_config(), root)
    sessions_root.mkdir(parents=True, exist_ok=True)
    default_real_dir.mkdir(parents=True, exist_ok=True)
    init_state(default_real_dir)

    folder_path, video_paths, calib_file = render_sidebar(default_real_dir, sessions_root)
    _route_first_run(calib_file)
    _step_picker()

    if st.session_state.ui_step == "setup":
        render_setup(video_paths, folder_path, calib_file)
    elif st.session_state.ui_step == "analyze":
        render_analyze(video_paths, folder_path, calib_file, sessions_root)
    else:
        render_result()
