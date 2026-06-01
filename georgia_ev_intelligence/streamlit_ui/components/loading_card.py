"""Animated loading card — replica of chat-interface-with-map/loading-steps.tsx.

The React version fakes four timed steps; here `render_step` is called from the
real `on_step` callback emitted by chat_service.answer (retrieval → dedup →
rerank → generation), so the card reflects actual pipeline progress.

All motion (bounce / pulse / fade-in) is declarative CSS keyframed in
theming/styles.py, so re-writing the placeholder's HTML on each step is enough
to animate it.
"""
from __future__ import annotations

import streamlit as st

# Label + description text taken verbatim from the React component, in backend
# step order. Index 0..3 = Searching KB / Retrieval / Reranking / Final answer.
STEPS = [
    ("Searching KB", "Scanning knowledge base..."),
    ("Retrieval", "Fetching relevant documents..."),
    ("Reranking", "Ranking by relevance..."),
    ("Final answer generation", "Composing response..."),
]

# Maps backend on_step event names → card index.
STEP_INDEX = {
    "retrieval": 0,
    "dedup": 1,
    "rerank": 2,
    "generation": 3,
}

# Inline lucide icons (Search, Database, ListFilter, Sparkles) so the card needs
# no icon font. White stroke to sit on the blue→indigo gradient tile.
_ICONS = [
    # Search
    "<circle cx='11' cy='11' r='8'></circle><path d='m21 21-4.3-4.3'></path>",
    # Database
    "<ellipse cx='12' cy='5' rx='9' ry='3'></ellipse>"
    "<path d='M3 5V19A9 3 0 0 0 21 19V5'></path><path d='M3 12A9 3 0 0 0 21 12'></path>",
    # ListFilter
    "<path d='M3 6h18'></path><path d='M7 12h10'></path><path d='M10 18h4'></path>",
    # Sparkles
    "<path d='M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 "
    "9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .962 0L14.063 8.5A2 2 0 0 0 "
    "15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 "
    "6.135a.5.5 0 0 1-.962 0z'></path>",
]


def _icon_svg(index: int) -> str:
    return (
        "<svg width='20' height='20' viewBox='0 0 24 24' fill='none' "
        "stroke='white' stroke-width='2' stroke-linecap='round' "
        f"stroke-linejoin='round'>{_ICONS[index]}</svg>"
    )


def card_html(active_index: int, completed_count: int) -> str:
    """Return the loading-card HTML for the given step state."""
    active_index = max(0, min(active_index, len(STEPS) - 1))
    label, description = STEPS[active_index]

    dots = "".join(
        f"<span class='lc-dot' style='animation-delay:{i * 150}ms'></span>"
        for i in range(3)
    )

    step_dots = ""
    for i in range(len(STEPS)):
        if i < completed_count:
            cls = "lc-stepdot lc-stepdot--done"
        elif i == active_index:
            cls = "lc-stepdot lc-stepdot--current"
        else:
            cls = "lc-stepdot lc-stepdot--pending"
        step_dots += f"<span class='{cls}'></span>"

    pct = ((completed_count + 0.5) / len(STEPS)) * 100

    return (
        "<div class='lc-wrap'>"
        "<div class='lc-card'>"
        "<div class='lc-body'>"
        "<div class='lc-iconwrap'>"
        f"<div class='lc-icon'>{_icon_svg(active_index)}</div>"
        "<div class='lc-ring'></div>"
        "</div>"
        "<div class='lc-text'>"
        f"<p class='lc-label'>{label}</p>"
        f"<p class='lc-desc'>{description}</p>"
        "</div>"
        f"<div class='lc-dots'>{dots}</div>"
        "</div>"
        "<div class='lc-footer'>"
        f"<div class='lc-stepdots'>{step_dots}</div>"
        f"<div class='lc-track'><div class='lc-fill' style='width:{pct:.1f}%'></div></div>"
        f"<p class='lc-count'>Step {active_index + 1} of {len(STEPS)}</p>"
        "</div>"
        "</div>"
        "</div>"
    )


def render_step(placeholder, active_index: int, completed_count: int) -> None:
    """Render/replace the loading card inside a passed st.empty() placeholder."""
    placeholder.markdown(card_html(active_index, completed_count), unsafe_allow_html=True)
