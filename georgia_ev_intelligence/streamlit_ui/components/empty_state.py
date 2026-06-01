"""Empty state — shown when there are no messages yet."""
from __future__ import annotations

import streamlit as st

from ..models.chat import SUGGESTED_QUESTIONS


def render(on_pick) -> None:
    """on_pick(question: str) is called when the user clicks a suggested card."""
    # Inline SVG bolt as the brand mark — inherits currentColor (the primary
    # tint set on .empty-logo) and stays crisp at any zoom level.
    st.markdown(
        """
        <div class="empty-shell">
            <div class="empty-logo">
                <svg width="30" height="30" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" stroke-width="2" stroke-linecap="round"
                     stroke-linejoin="round" aria-hidden="true">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                </svg>
            </div>
            <div class="empty-title">Georgia EV Supply Chain Intelligence</div>
            <div class="empty-subtitle">Ask questions about EV manufacturers, suppliers, and infrastructure in Georgia</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two columns on desktop; Streamlit stacks them to a single column on phones,
    # and the CSS card styling (st-key-suggest_*) handles hover/focus/wrapping.
    cols = st.columns(2)
    for index, question in enumerate(SUGGESTED_QUESTIONS):
        with cols[index % 2]:
            if st.button(
                f":material/bolt: {question}",
                key=f"suggest_{index}",
                use_container_width=True,
            ):
                on_pick(question)
