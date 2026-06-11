"""Render the message list with user/assistant bubbles + per-message copy."""
from __future__ import annotations

import html
import json
from datetime import datetime
from typing import List

import streamlit as st
import streamlit.components.v1 as components

from ..models.chat import Message
from ..models.source import Provenance
from ..state import ui_state
from . import sources_panel
from ._markdown import render_assistant_markdown


def _format_time(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M")
    except Exception:
        return ""


def _user_row(message: Message) -> str:
    # Single-line HTML, flush-left. Leading whitespace or blank lines inside the
    # block would trip CommonMark's indented-code-block rule and render the tags
    # as literal text in st.markdown(unsafe_allow_html=True).
    safe = html.escape(message.content).replace("\n", "<br>")
    timestamp = _format_time(message.timestamp)
    # No avatars — matches chat-interface-with-map (bubble is simply right-aligned).
    return (
        '<div class="chat-row chat-row--user">'
        '<div class="chat-col">'
        f'<div class="chat-bubble chat-bubble--user">{safe}</div>'
        f'<div class="chat-meta" style="text-align:right;">{timestamp}</div>'
        '</div>'
        '</div>'
    )


def _assistant_row(message: Message) -> str:
    rendered = render_assistant_markdown(message.content)
    timestamp = _format_time(message.timestamp)
    return (
        '<div class="chat-row">'
        '<div class="chat-col">'
        f'<div class="chat-bubble chat-bubble--assistant">{rendered}</div>'
        f'<div class="chat-meta">{timestamp}</div>'
        '</div>'
        '</div>'
    )


_CLIPBOARD_ICON = (
    "<svg width='14' height='14' viewBox='0 0 24 24' fill='none' "
    "stroke='currentColor' stroke-width='2' stroke-linecap='round' "
    "stroke-linejoin='round'><rect x='9' y='9' width='13' height='13' rx='2' "
    "ry='2'></rect><path d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 "
    "2 2v1'></path></svg>"
)


def _copy_control(text: str, align: str) -> None:
    """A small clipboard-icon button (with a 'copy' tooltip) inside an iframe.

    st.markdown strips onclick/script, so the copy handler must live in an
    iframe. The iframe can't read parent CSS vars, so neutral colors are used
    that read on both light and dark themes.
    """
    payload = json.dumps(text)
    justify = "flex-end" if align == "right" else "flex-start"
    components.html(
        f"""
        <style>
          body {{ margin: 0; }}
          .copy-wrap {{ display:flex; justify-content:{justify}; padding:0 6px; }}
          .copy-btn {{
            background: transparent; border: none; cursor: pointer;
            padding: 3px 6px; border-radius: 6px; color: #8b9bb3;
            display: inline-flex; align-items: center; gap: 4px;
            font: 600 11px/1 Inter, system-ui, sans-serif;
          }}
          .copy-btn:hover {{ background: rgba(127,127,127,0.16); color: #3863c4; }}
          .copy-btn.copied {{ color: #12897f; }}
        </style>
        <div class="copy-wrap">
          <button class="copy-btn" title="copy" onclick="copyText(this)">
            {_CLIPBOARD_ICON}<span class="lbl"></span>
          </button>
        </div>
        <script>
          const TEXT = {payload};
          function copyText(btn) {{
            navigator.clipboard.writeText(TEXT).then(() => {{
              const lbl = btn.querySelector('.lbl');
              btn.classList.add('copied');
              lbl.textContent = 'copied!';
              setTimeout(() => {{ btn.classList.remove('copied'); lbl.textContent=''; }}, 1200);
            }});
          }}
        </script>
        """,
        height=24,
    )


def render(messages: List[Message], provenance: Provenance) -> None:
    if not messages:
        return

    for message in messages:
        if message.role == "user":
            st.markdown(_user_row(message), unsafe_allow_html=True)
            _copy_control(message.content, align="right")
        else:
            st.markdown(_assistant_row(message), unsafe_allow_html=True)
            _copy_control(message.content, align="left")

    # Sources button and expanded provenance live directly below the latest
    # assistant answer inside the scrollable conversation.
    last = messages[-1]
    if last.role == "assistant" and provenance.has_content():
        open_now = ui_state.sources_panel_open()
        if open_now:
            label = "Hide Sources"
        elif provenance.count:
            label = f"View Sources ({provenance.count})"
        else:
            label = "View Sources"
        if st.button(label, key="chat_sources_toggle"):
            ui_state.toggle_sources_panel()
            st.rerun()
        if open_now:
            sources_panel.render(provenance)

    # Streamlit strips inline <script> from st.markdown, so the auto-scroll runs
    # inside an iframe; window.parent.document reaches the main Streamlit DOM.
    st.markdown("<div id='chat-bottom-anchor'></div>", unsafe_allow_html=True)
    components.html(
        """
        <script>
            const target = window.parent.document.getElementById('chat-bottom-anchor');
            const scroll = target && target.closest('.st-key-chat_scroll');
            if (scroll) scroll.scrollTo({top: scroll.scrollHeight, behavior: 'smooth'});
        </script>
        """,
        height=0,
    )
