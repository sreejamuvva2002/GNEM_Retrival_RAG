"""Settings panel — st.dialog overlay with theme + display toggles."""
from __future__ import annotations

import streamlit as st

from ..state import settings_state, ui_state


def render() -> None:
    if not ui_state.settings_open():
        return

    @st.dialog("Settings")
    def _dialog() -> None:
        current = settings_state.settings()

        st.markdown("##### Appearance")
        theme_options = ["☀ Light", "🌙 Dark"]
        current_theme = "🌙 Dark" if current.is_dark_mode else "☀ Light"
        picked_theme = st.segmented_control(
            "Theme",
            options=theme_options,
            default=current_theme,
            key="settings_theme",
            label_visibility="collapsed",
        )
        if picked_theme and picked_theme != current_theme:
            settings_state.set_dark_mode(picked_theme == "🌙 Dark")
            st.rerun()

        st.markdown("##### Display")
        show_citations = st.toggle(
            "Show Citations",
            value=current.show_citations,
            key="settings_show_citations",
            help="Display citation markers like [1][2] in responses.",
        )
        show_confidence = st.toggle(
            "Show Rank Score",
            value=current.show_confidence,
            key="settings_show_confidence",
            help="Display rank-based score bars on source cards.",
        )
        compact_mode = st.toggle(
            "Compact Mode",
            value=current.compact_mode,
            key="settings_compact",
            help="Reduce spacing and font size for higher density.",
        )

        settings_state.set_show_citations(show_citations)
        settings_state.set_show_confidence(show_confidence)
        settings_state.set_compact_mode(compact_mode)

        st.markdown("---")
        st.markdown(
            """
            **Georgia EV Supply Chain Intelligence**

            Domain-specific RAG chat over Georgia's electric vehicle ecosystem.
            Hybrid retrieval (BM25 + pgvector) + local LLM, with county-aware
            geospatial mapping.

            Version 1.0.0
            """
        )

        if st.button("Close", key="settings_close_btn", use_container_width=True):
            ui_state.set_settings_open(False)
            st.rerun()

    _dialog()
