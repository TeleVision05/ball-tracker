"""Step 3: speed, where it bounced, pitch map and replay for a delivery."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from ..plain_language import summarize_delivery
from ..session import list_deliveries
from ..user_messages import describe_error
from ..viz import draw_pitch_map, ensure_web_playable
from .logic import (
    NO_VALUE,
    default_delivery_index,
    details_markdown,
    overlay_video,
    source_video,
    speed_text,
)
from .state import goto


def _go_back_to_analyze() -> None:
    st.info("Analyze a ball first.")
    if st.button("Go to step 2"):
        goto("analyze")
        st.rerun()


def _pick_delivery(deliveries: list[dict]) -> dict:
    delivery_ids = [d["delivery_id"] for d in deliveries]
    default_id = st.session_state.last_delivery_id or delivery_ids[-1]

    if len(delivery_ids) > 1:
        selected = st.selectbox(
            "Previous deliveries",
            delivery_ids,
            index=default_delivery_index(delivery_ids, default_id),
        )
    else:
        selected = delivery_ids[0]
    return next(d for d in deliveries if d["delivery_id"] == selected)


def _replay(record: dict) -> None:
    """The overlay replay if it plays, else the original clip."""
    st.markdown("**Video replay**")
    overlay_path = overlay_video(record)
    played = False
    if overlay_path is not None:
        try:
            overlay_path = ensure_web_playable(overlay_path)
            st.video(overlay_path.read_bytes())
            played = True
        except Exception as exc:  # noqa: BLE001
            describe_error(exc, "preparing the replay video")
            st.warning(
                "We couldn't play the replay here, but your result is saved. "
                "Try again, or open the clip from the sessions folder."
            )
    if not played:
        src_path = source_video(record)
        if src_path is not None:
            st.video(src_path.read_bytes())
        else:
            st.error("No video found on disk.")


def _show_delivery(record: dict) -> None:
    metrics = record.get("metrics", {})
    summary = summarize_delivery(metrics)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Speed", speed_text(summary.get("speed_kmh")))
    with m2:
        st.markdown("**Where it bounced**")
        st.markdown(f"### {summary.get('bounce_text', NO_VALUE)}")

    col_map, col_media = st.columns(2)
    with col_map:
        st.markdown("**Pitch map**")
        fig = draw_pitch_map(metrics)
        st.pyplot(fig)
        plt.close(fig)

    with col_media:
        _replay(record)

    with st.expander("More details"):
        st.markdown(details_markdown(metrics))


def render_result() -> None:
    st.header("Result")
    session = st.session_state.session
    if session is None:
        _go_back_to_analyze()
        return
    deliveries = list_deliveries(session)
    if not deliveries:
        _go_back_to_analyze()
        return

    _show_delivery(_pick_delivery(deliveries))

    if st.button("Analyze another ball", type="primary"):
        st.session_state.last_delivery_id = None
        goto("analyze")
        st.rerun()
