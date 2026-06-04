"""Chat-panel header — dark GNEM badge (tooltip on hover) + title + subtitle.

Replica of the header inside chat-interface-with-map/chat-panel.tsx.
"""
from __future__ import annotations

import streamlit as st


def render() -> None:
    # The tooltip text lives in a sibling <span> rather than a `data-*` attr —
    # Streamlit's HTML sanitizer drops custom data-attributes, which made the
    # earlier `content: attr(data-tooltip)` CSS resolve to an empty string.
    st.markdown(
        """
        <div class="chat-header">
            <div class="gnem-logo-wrap" tabindex="0"
                 aria-label="Georgia Network for Electric Mobility">
                <span class="gnem-logo">GNEM</span>
                <span class="gnem-tooltip">Georgia Network for Electric Mobility</span>
            </div>
            <div class="chat-header__titles">
                <div class="chat-header__title">Chat Assistant</div>
                <div class="chat-header__subtitle">Georgia EV Supply Chain Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
