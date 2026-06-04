"""Left sidebar — chat history search + new-chat button + history list."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from ..state import chat_state


def _relative_time(timestamp: str) -> str:
    try:
        delta = datetime.now() - datetime.fromisoformat(timestamp)
    except Exception:
        return ""
    minutes = int(delta.total_seconds() // 60)
    if minutes < 60:
        return f"{max(minutes, 1)}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return "Yesterday" if days == 1 else f"{days}d ago"


def render() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-heading">
                <span class="sidebar-heading__badge">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
                         stroke="currentColor" stroke-width="2" stroke-linecap="round"
                         stroke-linejoin="round" aria-hidden="true">
                        <path d="M3 3v5h5"></path>
                        <path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"></path>
                        <path d="M12 7v5l4 2"></path>
                    </svg>
                </span>
                <span class="sidebar-heading__label">History</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(":material/add: New Chat", key="sb_new_chat", use_container_width=True, type="secondary"):
            chat_state.start_new_chat()
            st.rerun()

        search = st.text_input(
            "Search chats",
            key="sb_search",
            placeholder="Search chats...",
            label_visibility="collapsed",
        )

        entries = chat_state.history()
        if search:
            needle = search.lower()
            entries = [
                e for e in entries
                if needle in (e.title or "").lower() or needle in (e.preview or "").lower()
            ]

        current_id = chat_state.current_chat_id()
        if not entries:
            st.markdown(
                "<p class='sidebar-empty'>No chat history yet</p>",
                unsafe_allow_html=True,
            )
            return

        for entry in entries:
            is_active = entry.id == current_id
            relative = _relative_time(entry.timestamp)
            title_text = entry.title if len(entry.title) <= 34 else entry.title[:33] + "…"
            select_col, delete_col = st.columns([0.85, 0.15])
            with select_col:
                # Active conversation is shown with the filled (primary) button —
                # a non-color cue (fill) on top of the brand color.
                if st.button(
                    f":material/chat_bubble: {title_text}",
                    key=f"sb_open_{entry.id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    chat_state.set_current_chat_id(entry.id)
                    st.rerun()
            with delete_col:
                if st.button(
                    ":material/delete:",
                    key=f"sb_del_{entry.id}",
                    use_container_width=True,
                    help="Delete chat",
                ):
                    chat_state.remove_history_entry(entry.id)
                    st.rerun()
            # The title button already shows the question — the caption only adds
            # metadata (no repeated question text).
            st.caption(f"{relative} · {entry.message_count} msgs")

        st.markdown(
            "<p class='sidebar-footer'>Georgia EV Intelligence</p>",
            unsafe_allow_html=True,
        )
