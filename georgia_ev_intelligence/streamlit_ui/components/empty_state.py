"""Empty state — "Start a conversation" plus quick-pick validated questions.

The centered hero mirrors chat-interface-with-map. Below it we offer the
human-validated question set (``data/questions_50.csv``) as quick prompts: a few
one-click buttons and an expander with the full list. ``on_pick(question)`` is
invoked when the user selects one (wired to the chat submit handler), while
free-text questions still go through the chat input box.
"""
from __future__ import annotations

import streamlit as st

_QUICK_PICK_COUNT = 6


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

    if on_pick is None:
        return

    # Lazy import keeps the heavy services module out of the import path until the
    # empty state actually renders.
    from ..services.cache import get_example_questions

    questions = get_example_questions()
    if not questions:
        return

    st.markdown(
        "<div class='empty-subtitle' style='margin-top:0.5rem'>"
        "Or try one of the 50 human-validated questions:</div>",
        unsafe_allow_html=True,
    )

    for qid, question in questions[:_QUICK_PICK_COUNT]:
        if st.button(question, key=f"ex_btn_{qid}", use_container_width=True):
            on_pick(question)

    if len(questions) > _QUICK_PICK_COUNT:
        with st.expander(f"Browse all {len(questions)} validated questions"):
            labels = [f"{qid} — {question}" for qid, question in questions]
            choice = st.selectbox("Pick a question", labels, key="ex_select")
            if st.button("Ask this question", key="ex_select_run"):
                on_pick(questions[labels.index(choice)][1])
