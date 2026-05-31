"""Dashboard widgets — 4 stat cards above the chat area."""
from __future__ import annotations

import streamlit as st

from ..state import chat_state


def _widget_card(icon: str, value: str, label: str) -> str:
    return f"""
    <div class="widget-card">
        <div class="widget-card__icon">{icon}</div>
        <div class="widget-card__value">{value}</div>
        <div class="widget-card__label">{label}</div>
    </div>
    """


def render() -> None:
    dispatch = chat_state.last_dispatch()
    if dispatch is None:
        sources_count = 0
        mapped_count = 0
        county_coverage = 0
        focus = "—"
    else:
        sources_count = len(dispatch.chat.parent_contexts)
        mapped_count = len(dispatch.map.records)
        county_coverage = dispatch.map.context.county_coverage_count
        focus = dispatch.map.context.focus_label or dispatch.map.context.map_mode.title()

    columns = st.columns(4)
    payloads = [
        ("📄", str(mapped_count), "Mapped Companies"),
        ("📚", str(sources_count), "Sources Retrieved"),
        ("🗺", str(county_coverage), "County Coverage"),
        ("🎯", focus, "Focus"),
    ]
    for col, (icon, value, label) in zip(columns, payloads):
        with col:
            st.markdown(_widget_card(icon, value, label), unsafe_allow_html=True)
