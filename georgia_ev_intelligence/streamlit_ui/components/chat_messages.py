"""Render the message list with user/assistant bubbles and citation chips."""
from __future__ import annotations

import html
from datetime import datetime
from typing import List

import streamlit as st
import streamlit.components.v1 as components

from ..models.chat import Message
from ..models.source import SourceViewModel
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
    return (
        '<div class="chat-row chat-row--user">'
        '<span class="chat-avatar chat-avatar--user">U</span>'
        '<div>'
        f'<div class="chat-bubble chat-bubble--user">{safe}</div>'
        f'<div class="chat-meta" style="text-align:right;">{timestamp}</div>'
        '</div>'
        '</div>'
    )


def _assistant_row(message: Message, sources_by_id: dict) -> str:
    rendered = render_assistant_markdown(message.content)
    timestamp = _format_time(message.timestamp)

    chips = ""
    if message.source_ids:
        chips_html = "".join(
            (
                "<span class='source-type-pill' style='background:var(--citation-bg); color:var(--citation-text);'>"
                f"{i + 1} · {html.escape((sources_by_id.get(sid).title if sources_by_id.get(sid) else sid)[:48])}"
                "</span>"
            )
            for i, sid in enumerate(message.source_ids[:5])
        )
        chips = (
            "<div style='margin-top:0.5rem; display:flex; flex-wrap:wrap; gap:0.3rem;'>"
            f"{chips_html}</div>"
        )

    return (
        '<div class="chat-row">'
        '<span class="chat-avatar chat-avatar--assistant">●</span>'
        '<div>'
        f'<div class="chat-bubble chat-bubble--assistant">{rendered}</div>'
        f'{chips}'
        f'<div class="chat-meta">{timestamp}</div>'
        '</div>'
        '</div>'
    )


def render(messages: List[Message], sources: List[SourceViewModel]) -> None:
    sources_by_id = {s.id: s for s in sources}
    html_parts: list[str] = []
    for message in messages:
        if message.role == "user":
            html_parts.append(_user_row(message))
        else:
            html_parts.append(_assistant_row(message, sources_by_id))

    if not html_parts:
        return

    container = "".join(html_parts) + "<div id='chat-bottom-anchor'></div>"
    # Container kept as one continuous HTML blob (no embedded newlines that could
    # bleed indentation past markdown's HTML-block detection).
    st.markdown(
        "<div style='padding:0.4rem 0.6rem 0.8rem;'>" + container + "</div>",
        unsafe_allow_html=True,
    )
    # Streamlit strips inline <script> tags from st.markdown, so the scroll has to
    # run inside an iframe via components.v1.html. window.parent.document reaches
    # the main Streamlit DOM where the anchor div was rendered.
    components.html(
        """
        <script>
            const target = window.parent.document.getElementById('chat-bottom-anchor');
            if (target) target.scrollIntoView({behavior: 'smooth', block: 'end'});
        </script>
        """,
        height=0,
    )
