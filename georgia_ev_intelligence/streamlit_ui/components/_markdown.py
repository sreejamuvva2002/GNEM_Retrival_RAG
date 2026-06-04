"""Private helpers: render assistant markdown with [n] citation chips inline.

This mirrors the React MarkdownRenderer in components/chat/ChatMessage.jsx —
we accept the same lightweight markdown subset (bold, headers, bullet/numbered
lists, paragraphs) and render [1][2][3] as styled inline chips.
"""
from __future__ import annotations

import html
import re
from typing import List

_CITATION_RE = re.compile(r"\[(\d+)\]")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")


def _replace_citations(safe_text: str) -> str:
    return _CITATION_RE.sub(
        lambda m: f"<span class='citation-chip'>{html.escape(m.group(1))}</span>",
        safe_text,
    )


def _render_bold(safe_text: str) -> str:
    return _BOLD_RE.sub(lambda m: f"<strong>{m.group(1)}</strong>", safe_text)


def render_assistant_markdown(text: str) -> str:
    """Return safe HTML for an assistant response."""
    if not text:
        return "<p class='muted'>(no answer)</p>"

    lines = text.splitlines()
    html_parts: List[str] = []
    current_list: List[str] | None = None
    list_kind: str | None = None  # "ul" or "ol"

    def flush_list() -> None:
        nonlocal current_list, list_kind
        if current_list and list_kind:
            html_parts.append(f"<{list_kind}>" + "".join(f"<li>{item}</li>" for item in current_list) + f"</{list_kind}>")
        current_list = None
        list_kind = None

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            flush_list()
            continue

        safe = _render_bold(_replace_citations(html.escape(line)))

        # Bullet list
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            item = _render_bold(_replace_citations(html.escape(bullet.group(1))))
            if list_kind != "ul":
                flush_list()
                list_kind = "ul"
                current_list = []
            current_list.append(item)
            continue

        # Numbered list
        numbered = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if numbered:
            item = _render_bold(_replace_citations(html.escape(numbered.group(1))))
            if list_kind != "ol":
                flush_list()
                list_kind = "ol"
                current_list = []
            current_list.append(item)
            continue

        # Heading (line wrapped in **...**)
        heading_match = re.match(r"^\s*\*\*(.+?)\*\*\s*:?$", line)
        if heading_match:
            flush_list()
            heading_text = _replace_citations(html.escape(heading_match.group(1)))
            html_parts.append(f"<h4>{heading_text}</h4>")
            continue

        flush_list()
        html_parts.append(f"<p>{safe}</p>")

    flush_list()
    return "".join(html_parts)
