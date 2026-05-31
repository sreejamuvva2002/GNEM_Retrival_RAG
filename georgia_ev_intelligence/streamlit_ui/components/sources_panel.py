"""Right-side sources panel — shows ParentContext-derived SourceViewModel cards."""
from __future__ import annotations

import html
from typing import List

import streamlit as st

from ..models.chat import Settings
from ..models.source import SourceViewModel
from ..state import ui_state
from ..theming.colors import SOURCE_TYPE_COLORS


def _type_label(source_type: str) -> str:
    return source_type.replace("_", " ").title()


def _type_pill(source_type: str) -> str:
    fg, bg = SOURCE_TYPE_COLORS.get(source_type, SOURCE_TYPE_COLORS["unknown"])
    label = html.escape(_type_label(source_type))
    return f"<span class='source-type-pill' style='background:{bg}; color:{fg};'>● {label}</span>"


def _source_card_html(source: SourceViewModel, settings: Settings, total: int) -> str:
    # Keep the entire card on a single line with no leading whitespace — markdown
    # treats 4-space-indented lines as code blocks (see _assistant_row in
    # chat_messages.py for the same lesson).
    location_part = (
        f" · 📍 {html.escape(source.location_name)}" if source.location_name else ""
    )
    rank_block = ""
    if settings.show_confidence:
        pct = int(round(source.rank_score * 100))
        rank_block = (
            f'<div class="source-rank-bar"><div class="source-rank-bar__fill" style="width:{pct}%;"></div></div>'
            f'<div class="source-rank-label">Position rank · #{source.rank} of {total}</div>'
        )

    return (
        '<div class="source-card">'
        f"{_type_pill(source.source_type)}"
        f'<h4 class="source-card__title">{html.escape(source.title)}</h4>'
        f'<p class="source-card__meta">#{source.rank} · {html.escape(source.record_id)}{location_part}</p>'
        f'<p class="source-card__snippet">{html.escape(source.snippet)}</p>'
        f"{rank_block}"
        '</div>'
    )


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

    total = len(sources)
    for source in sources:
        st.markdown(_source_card_html(source, settings, total), unsafe_allow_html=True)
        with st.expander("View full chunk text"):
            st.code(source.parent_chunk_text or "(empty)", language=None)
