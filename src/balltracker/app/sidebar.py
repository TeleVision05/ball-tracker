"""Sidebar: video upload and folder, camera-setup status, and session loading."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..session import load_session
from .logic import list_videos

NEW_SESSION = "(new session)"


def _videos_section(default_real_dir: Path) -> tuple[Path, list[Path]]:
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

    video_paths = list_videos(folder_path)
    st.caption(f"{len(video_paths)} video(s) available")
    return folder_path, video_paths


def _session_section(sessions_root: Path) -> None:
    st.subheader("Session")
    if st.session_state.session is not None:
        st.caption(f"Active: `{st.session_state.session.id}`")
    existing = sorted(sessions_root.glob("*/session.json"))
    existing_labels = [NEW_SESSION] + [p.parent.name for p in existing]
    pick = st.selectbox("Load an existing session", existing_labels, index=0)
    if pick != NEW_SESSION and st.button("Load"):
        st.session_state.session = load_session(sessions_root / pick)
        st.success(f"Loaded {pick}")


def render_sidebar(default_real_dir: Path, sessions_root: Path) -> tuple[Path, list[Path], Path]:
    """Draw the sidebar; return the video folder, its videos and the calibration file."""
    with st.sidebar:
        folder_path, video_paths = _videos_section(default_real_dir)

        st.session_state.calib_path = str(folder_path / "session.calib.json")
        st.divider()
        calib_file = Path(st.session_state.calib_path).expanduser()
        if calib_file.exists():
            st.success("Camera setup saved")
        else:
            st.info("Camera setup needed")

        st.divider()
        _session_section(sessions_root)
    return folder_path, video_paths, calib_file
