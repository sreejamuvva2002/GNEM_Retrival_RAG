"""ChatApp orchestrator — the only file Streamlit's CLI runs directly.

Responsibilities:
  * Initialize session state.
  * Inject CSS for the active theme.
  * Render header + sidebar + dashboard + main view (chat/split/map) + sources panel.
  * Drive a single submitted query through QueryDispatcher and update state.

Business logic lives in services/ + spatial/. This module only composes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import streamlit as st

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from georgia_ev_intelligence.streamlit_ui.components import (
    chat_messages,
    empty_state,
    header,
    loading_card,
    map_view,
    resizable_split,
    sources_panel,
)
from georgia_ev_intelligence.streamlit_ui.models.chat import ChatMemory, ChatTurnMetadata
from georgia_ev_intelligence.streamlit_ui.models.source import SourceViewModel
from georgia_ev_intelligence.streamlit_ui.services.cache import (
    dispatch_query,
    get_xlsx_lookup,
    summarize_chat_memory,
)
from georgia_ev_intelligence.streamlit_ui.state import chat_state, settings_state, ui_state
from georgia_ev_intelligence.streamlit_ui.theming.styles import inject_styles


def _bootstrap_state() -> None:
    chat_state.initialize()
    ui_state.initialize()
    settings_state.initialize()


def _enrich_sources() -> List[SourceViewModel]:
    dispatch = chat_state.last_dispatch()
    if dispatch is None:
        return []
    lookup = get_xlsx_lookup()
    total = len(dispatch.chat.parent_contexts)
    return [
        SourceViewModel.from_parent_context(parent, rank=i + 1, total=total, xlsx_lookup=lookup)
        for i, parent in enumerate(dispatch.chat.parent_contexts)
    ]


def _compact_conversation_summary() -> None:
    """Summarize older completed turns while preserving the last 4 turns raw."""
    batch, next_cursor = chat_state.summary_compaction_batch(recent_turns=4)
    if not batch or next_cursor <= chat_state.summary_cursor():
        return

    try:
        updated_summary = summarize_chat_memory(chat_state.conversation_summary(), batch)
    except Exception:
        return

    chat_state.set_conversation_summary(updated_summary)
    chat_state.set_summary_cursor(next_cursor)


def _run_pending_query(
    query: str,
    chat_memory: ChatMemory | None,
    placeholder,
) -> None:
    """Run dispatch for an already-shown user message, animating the loading card.

    The real on_step events (retrieval → dedup → rerank → generation) drive the
    React-style loading card; on completion we push the answer and rerun.
    """
    loading_card.render_step(placeholder, active_index=0, completed_count=0)

    def _on_step(name: str) -> None:
        idx = loading_card.STEP_INDEX.get(name)
        if idx is None:
            return
        # The current step is "active"; every earlier step is complete.
        loading_card.render_step(placeholder, active_index=idx, completed_count=idx)

    try:
        dispatch = dispatch_query(query, chat_memory=chat_memory, _on_step=_on_step)
    except Exception as exc:
        placeholder.empty()
        ui_state.clear_pending_query()
        chat_state.append_message(
            chat_state.make_assistant_message(
                f"⚠️ Could not reach the retrieval pipeline. Please retry once the "
                f"backend is ready. ({exc})",
                source_ids=[],
            )
        )
        st.rerun()
        return

    placeholder.empty()

    source_ids: list[str] = []
    if dispatch.chat.error and not dispatch.chat.answer:
        assistant_content = f"⚠️ {dispatch.chat.error}"
        chat_state.append_message(
            chat_state.make_assistant_message(assistant_content, source_ids=[])
        )
    else:
        source_ids = [p.record_id for p in dispatch.chat.parent_contexts]
        assistant_content = dispatch.chat.answer or "(no answer returned)"
        chat_state.append_message(
            chat_state.make_assistant_message(
                assistant_content,
                source_ids=source_ids,
            )
        )

    chat_state.set_last_dispatch(dispatch)
    chat_state.append_turn_metadata(
        ChatTurnMetadata(
            original_query=query,
            effective_query=dispatch.chat.effective_query or query,
            history_used=dispatch.chat.history_used,
            source_ids=source_ids,
            trace=dict(dispatch.chat.trace or {}),
        )
    )
    _compact_conversation_summary()
    ui_state.clear_pending_query()
    st.rerun()


def _render_chat_pane(sources: List[SourceViewModel]) -> None:
    messages = chat_state.messages()
    if not messages:
        empty_state.render(on_pick=_handle_submit)
    else:
        chat_messages.render(messages, sources)


def _render_map_pane(sources: List[SourceViewModel], is_dark: bool, height: int = 600) -> None:
    dispatch = chat_state.last_dispatch()
    if dispatch is None:
        # Empty map + "Ask a question…" overlay, matching the React empty state.
        map_view.render([], {}, is_dark=is_dark, height=height)
        return

    # Build markers straight from the cited sources so every located source has a
    # pin (the spatial map.records are a different, often-filtered set, which left
    # most located sources pinless). Sources without a usable location yield no
    # marker — matching the no-location icon in the sources panel.
    records = sources_panel.map_records(sources)
    map_view.render(records, dispatch.map.context.to_dict(), is_dark=is_dark, height=height)


def _handle_submit(query: str) -> None:
    # Show the user bubble immediately; the answer is produced on the next run
    # (see _run_pending_query) so the loading card renders under the message.
    chat_memory = chat_state.rag_memory(recent_turns=4)
    chat_state.append_message(chat_state.make_user_message(query))
    ui_state.set_pending_chat_memory(chat_memory)
    ui_state.set_pending_query(query)
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Georgia EV Supply Chain Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _bootstrap_state()
    s = settings_state.settings()
    inject_styles(is_dark=False, compact=False)

    sources = _enrich_sources()
    show_sources = ui_state.sources_panel_open() and bool(sources)
    pending = ui_state.pending_query()
    pending_memory = ui_state.pending_chat_memory()

    # 50/50 split: chat left, map (+ stacked sources) right — matches React.
    # Heights (full-viewport columns, scrollable messages, full/50-50 map) are
    # applied client-side by resizable_split.render().
    chat_col, map_col = st.columns([0.5, 0.5])

    loading_ph = None
    with chat_col:
        resizable_split.anchor()  # sentinel for the divider/layout script
        header.render()
        # Scrollable messages region; the inline chat input below it is pinned to
        # the bottom of the column (the JS makes this container flex:1).
        with st.container(key="chat_scroll"):
            _render_chat_pane(sources)
            if pending:
                # Loading card renders here (under the user bubble); filled after
                # the map renders so the right pane shows during the dispatch.
                loading_ph = st.empty()
        # Inline (non-docked) search bar — lives inside the left column.
        submitted = st.chat_input("Ask about EV companies in Georgia...", key="chat_input")
    with map_col:
        _render_map_pane(sources, is_dark=False)  # map wraps itself in st.container(key="gnem_map")
        if show_sources:
            with st.container(key="gnem_sources"):
                sources_panel.render(sources, s)

    # Draggable divider + full-height flex layout — all client-side (no rerun).
    resizable_split.render()

    if pending and loading_ph is not None:
        _run_pending_query(pending, pending_memory, loading_ph)

    if submitted:
        _handle_submit(submitted)


main()
