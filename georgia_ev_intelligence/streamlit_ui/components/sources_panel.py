"""Right-side sources panel — scrollable list; name-only with per-source expand."""
from __future__ import annotations

from typing import List

import streamlit as st

from ..models.chat import Settings
from ..models.source import SourceViewModel
from ..state import ui_state


def render(sources: List[SourceViewModel], settings: Settings) -> None:
    st.markdown(
        f"""
        <div class="sources-header">
            <div>
                <div class="sources-header__title">Sources</div>
                <div class="sources-header__subtitle">
                    {len(sources)} source{'s' if len(sources) != 1 else ''} found
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("✕ Close panel", key="sources_close", use_container_width=True):
        ui_state.set_sources_panel_open(False)
        st.rerun()

    if not sources:
        st.info("Submit a question to see retrieved sources here.")
        return

    # Native fixed-height container = scrollable list. Each source shows only the
    # company/county name; expanding it reveals just the full chunk text (the old
    # snippet was a truncated copy of this, so it duplicated the content).
    with st.container(height=420):
        for source in sources:
            name = source.title or source.record_id or "Source"
            with st.expander(name):
                st.code(source.parent_chunk_text or "(empty)", language=None)
