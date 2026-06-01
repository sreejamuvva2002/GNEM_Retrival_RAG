"""Empty state — "Start a conversation", matching chat-interface-with-map.

The React app shows only a centered icon tile + heading + subtitle (no
suggested-question cards), so this mirrors that exactly. `on_pick` is accepted
for call-site compatibility but unused.
"""
from __future__ import annotations

import streamlit as st


def render(on_pick=None) -> None:
    st.markdown(
        """
        <div class="empty-shell">
            <div class="empty-logo">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" stroke-width="2" stroke-linecap="round"
                     stroke-linejoin="round" aria-hidden="true">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
            </div>
            <div class="empty-title">Start a conversation</div>
            <div class="empty-subtitle">Ask me about companies in Georgia's automotive and EV supply chain.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
