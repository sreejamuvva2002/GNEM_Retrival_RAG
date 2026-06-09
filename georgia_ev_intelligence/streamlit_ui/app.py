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
from georgia_ev_intelligence.streamlit_ui.models.source import Provenance, SourceViewModel
from georgia_ev_intelligence.streamlit_ui.services.cache import (
    dispatch_query_cached,
    get_xlsx_lookup,
)
from georgia_ev_intelligence.streamlit_ui.services.chat_service import (
    extract_cited_company_names,
)
from georgia_ev_intelligence.streamlit_ui.services.map_service import (
    filter_records_to_companies,
)
from georgia_ev_intelligence.streamlit_ui.state import chat_state, settings_state, ui_state
from georgia_ev_intelligence.streamlit_ui.theming.styles import inject_styles


def _bootstrap_state() -> None:
    chat_state.initialize()
    ui_state.initialize()
    settings_state.initialize()


def _build_provenance() -> Provenance:
    """Assemble the answer's sources: evidence records + the executed query.

    SQL-backed routes carry their grounding rows in ``chat.evidence_rows`` (one
    KB record each) — these become source cards. Aggregate routes carry group
    rows instead, shown alongside the SQL. Document routes still enrich the
    parent contexts via the xlsx lookup. The SQL itself is method, not a source,
    so it rides along in ``sql_queries`` for a separate panel section.
    """
    dispatch = chat_state.last_dispatch()
    if dispatch is None:
        return Provenance()
    chat = dispatch.chat

    if chat.evidence_kind == "records" and chat.evidence_rows:
        total = len(chat.evidence_rows)
        sources = [
            SourceViewModel.from_evidence_row(row, rank=i + 1, total=total)
            for i, row in enumerate(chat.evidence_rows)
        ]
        return Provenance(sources=sources, sql_queries=chat.sql_queries, kind="records")

    if chat.evidence_kind in ("groups", "count"):
        return Provenance(
            group_rows=chat.evidence_rows,
            sql_queries=chat.sql_queries,
            kind=chat.evidence_kind,
        )

    # Document / hybrid route: enrich parent contexts from the workbook.
    lookup = get_xlsx_lookup()
    total = len(chat.parent_contexts)
    sources = [
        SourceViewModel.from_parent_context(parent, rank=i + 1, total=total, xlsx_lookup=lookup)
        for i, parent in enumerate(chat.parent_contexts)
    ]
    return Provenance(sources=sources, sql_queries=chat.sql_queries, kind="documents")


def _record_history() -> None:
    """Add a history entry for the first user turn of a fresh chat."""
    msgs = chat_state.messages()
    if not msgs:
        return
    first_user = next((m for m in msgs if m.role == "user"), None)
    if first_user and chat_state.current_chat_id() is None:
        preview = first_user.content[:60]
        title = preview if len(preview) < 36 else preview[:34] + "..."
        chat_state.add_history_entry(title=title, preview=preview, message_count=len(msgs))


def _run_pending_query(query: str, placeholder) -> None:
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

    history_tuples = tuple((m.role, m.content) for m in chat_state.messages()[:-1])
    try:
        dispatch = dispatch_query_cached(query, history=history_tuples, _on_step=_on_step)
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

    if dispatch.chat.error and not dispatch.chat.answer:
        chat_state.append_message(
            chat_state.make_assistant_message(f"⚠️ {dispatch.chat.error}", source_ids=[])
        )
    else:
        chat_state.append_message(
            chat_state.make_assistant_message(
                dispatch.chat.answer or "(no answer returned)",
                source_ids=[p.record_id for p in dispatch.chat.parent_contexts],
            )
        )

    chat_state.set_last_dispatch(dispatch)
    _record_history()
    ui_state.clear_pending_query()
    st.rerun()


def _render_chat_pane(provenance: Provenance) -> None:
    messages = chat_state.messages()
    if not messages:
        empty_state.render(on_pick=_handle_submit)
    else:
        chat_messages.render(messages, provenance)


def _render_map_pane(is_dark: bool, height: int = 600) -> None:
    dispatch = chat_state.last_dispatch()
    if dispatch is None:
        # Empty map + "Ask a question…" overlay, matching the React empty state.
        map_view.render([], {}, is_dark=is_dark, height=height)
        return

    # Prefer the geocoded companies the answer was actually built from (route
    # executor evidence). These are the exact companies in the answer, so the
    # map and the text never diverge. Fall back to the separate map pipeline +
    # cited-company filter only when the answer carries no map records.
    if dispatch.chat.map_records:
        records = dispatch.chat.map_records
    else:
        cited = extract_cited_company_names(dispatch.chat.parent_contexts)
        records = filter_records_to_companies(dispatch.map.records, cited)
    map_view.render(records, dispatch.map.context.to_dict(), is_dark=is_dark, height=height)


def _handle_submit(query: str) -> None:
    # Show the user bubble immediately; the answer is produced on the next run
    # (see _run_pending_query) so the loading card renders under the message.
    chat_state.append_message(chat_state.make_user_message(query))
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

    provenance = _build_provenance()
    show_sources = ui_state.sources_panel_open() and provenance.has_content()
    pending = ui_state.pending_query()

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
            _render_chat_pane(provenance)
            if pending:
                # Loading card renders here (under the user bubble); filled after
                # the map renders so the right pane shows during the dispatch.
                loading_ph = st.empty()
        # Inline (non-docked) search bar — lives inside the left column.
        submitted = st.chat_input("Ask about EV companies in Georgia...", key="chat_input")
    with map_col:
        _render_map_pane(is_dark=False)  # map wraps itself in st.container(key="gnem_map")
        if show_sources:
            with st.container(key="gnem_sources"):
                sources_panel.render(provenance, s)

    # Draggable divider + full-height flex layout — all client-side (no rerun).
    resizable_split.render()

    if pending and loading_ph is not None:
        _run_pending_query(pending, loading_ph)

    if submitted:
        _handle_submit(submitted)


main()
