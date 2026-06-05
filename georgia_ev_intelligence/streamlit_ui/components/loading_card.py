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

# Label + description text in self-healing loop execution order. The open-loop
# path emits a subset (retrieval/dedup/rerank/generation); the self-healing path
# adds decompose/judge/verify/retry. With retries the active step can move
# backwards (re-searching) — that's intentional and reflects real progress.
STEPS = [
    ("Planning sub-queries", "Breaking the question into parts..."),
    ("Searching KB", "Scanning knowledge base..."),
    ("Reranking", "Ranking by relevance..."),
    ("Judging evidence", "Checking the retrieval is relevant..."),
    ("Generating answer", "Composing response..."),
    ("Verifying answer", "Checking the answer is grounded..."),
    ("Re-searching", "Healing retrieval and retrying..."),
]

# Maps backend on_step event names → card index.
STEP_INDEX = {
    "decompose": 0,
    "retrieval": 1,
    "dedup": 2,
    "rerank": 2,
    "judge": 3,
    "generation": 4,
    "verify": 5,
    "retry": 6,
}

# Inline lucide icons (one per STEPS entry, same order) so the card needs no icon
# font. White stroke to sit on the blue→indigo gradient tile.
_ICONS = [
    # GitBranch (decompose)
    "<line x1='6' x2='6' y1='3' y2='15'></line><circle cx='18' cy='6' r='3'></circle>"
    "<circle cx='6' cy='18' r='3'></circle><path d='M18 9a9 9 0 0 1-9 9'></path>",
    # Search (retrieval)
    "<circle cx='11' cy='11' r='8'></circle><path d='m21 21-4.3-4.3'></path>",
    # ListFilter (rerank)
    "<path d='M3 6h18'></path><path d='M7 12h10'></path><path d='M10 18h4'></path>",
    # CheckCheck (judge)
    "<path d='M18 6 7 17l-5-5'></path><path d='m22 10-7.5 7.5L13 16'></path>",
    # Sparkles (generation)
    "<path d='M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 "
    "9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .962 0L14.063 8.5A2 2 0 0 0 "
    "15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 "
    "6.135a.5.5 0 0 1-.962 0z'></path>",
    # ShieldCheck (verify)
    "<path d='M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 "
    "1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 "
    "1 0 0 1 1 1z'></path><path d='m9 12 2 2 4-4'></path>",
    # RotateCw (retry)
    "<path d='M21 12a9 9 0 1 1-3-6.7L21 8'></path><path d='M21 3v5h-5'></path>",
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
