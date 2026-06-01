"""Empty state — shown when there are no messages yet."""
from __future__ import annotations

import streamlit as st

from ..models.chat import SUGGESTED_QUESTIONS


def render(on_pick) -> None:
    """on_pick(question: str) is called when the user clicks a suggested card."""
    st.markdown(
        """
        <div class="empty-shell">
            <div class="empty-logo">✨</div>
            <div class="empty-title">Georgia EV Supply Chain Intelligence</div>
            <div class="empty-subtitle">Ask questions about EV manufacturers, suppliers, and infrastructure in Georgia</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for index, question in enumerate(SUGGESTED_QUESTIONS):
        with cols[index % 2]:
            if st.button(
                f"⚡ {question}",
                key=f"suggest_{index}",
                use_container_width=True,
            ):
                on_pick(question)
