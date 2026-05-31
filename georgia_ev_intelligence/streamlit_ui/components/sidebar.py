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
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.6rem;">
                <span style="width:32px; height:32px; border-radius:9px; background: var(--primary);
                             color: var(--primary-fg); display:inline-flex; align-items:center;
                             justify-content:center; font-weight:800;">⌕</span>
                <strong style="font-size:0.95rem;">History</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("+ New Chat", key="sb_new_chat", use_container_width=True, type="secondary"):
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
                "<p style='color: var(--muted-fg); font-size:0.8rem; text-align:center; margin-top:1.2rem;'>"
                "No chat history yet</p>",
                unsafe_allow_html=True,
            )
            return

        for entry in entries:
            is_active = entry.id == current_id
            relative = _relative_time(entry.timestamp)
            title_text = entry.title if len(entry.title) <= 34 else entry.title[:33] + "…"
            select_col, delete_col = st.columns([0.85, 0.15])
            with select_col:
                if st.button(
                    f"{'●' if is_active else '○'}  {title_text}",
                    key=f"sb_open_{entry.id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    chat_state.set_current_chat_id(entry.id)
                    st.rerun()
            with delete_col:
                if st.button("✕", key=f"sb_del_{entry.id}", use_container_width=True):
                    chat_state.remove_history_entry(entry.id)
                    st.rerun()
            preview_text = entry.preview if len(entry.preview) <= 48 else entry.preview[:47] + "…"
            st.caption(f"{preview_text} · {relative} · {entry.message_count} msgs")

        st.markdown(
            "<p style='color: var(--muted-fg); font-size:0.7rem; text-align:center; margin-top:1.2rem;'>"
            "Georgia EV Intelligence</p>",
            unsafe_allow_html=True,
        )
