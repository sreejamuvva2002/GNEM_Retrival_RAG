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
    map_view,
    settings_panel,
    sidebar,
    sources_panel,
)  # dashboard intentionally not imported — widget row commented out (see main()).
from georgia_ev_intelligence.streamlit_ui.models.source import SourceViewModel
from georgia_ev_intelligence.streamlit_ui.services.cache import (
    baseline_map_payload,
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


_STEP_SEQUENCE = [
    ("retrieval", "Retrieving relevant sources"),
    ("dedup", "Deduplicating results"),
    ("rerank", "Reranking results"),
    ("generation", "Generating the final answer"),
]


def _process_query(query: str) -> None:
    """Push user message → run dispatch with a 4-step status → push answer."""
    chat_state.append_message(chat_state.make_user_message(query))

    try:
        with st.status("Working on your question…", expanded=True) as status:
            labels = {key: label for key, label in _STEP_SEQUENCE}
            placeholders = {key: st.empty() for key, _ in _STEP_SEQUENCE}
            for key, label in _STEP_SEQUENCE:
                placeholders[key].markdown(f"⚪ {label}")

            done: list[str] = []

            def _on_step(name: str) -> None:
                # Mark every earlier step done, and the current one as running.
                for prev in done:
                    placeholders[prev].markdown(f"✅ {labels[prev]}")
                if name in placeholders:
                    placeholders[name].markdown(f"⏳ {labels[name]}")
                    done.append(name)

            dispatch = dispatch_query_cached(query, _on_step=_on_step)

            for key, label in _STEP_SEQUENCE:
                placeholders[key].markdown(f"✅ {label}")
            status.update(label="Done", state="complete", expanded=False)
    except Exception as exc:
        st.error(f"Backend unavailable: {exc}")
        chat_state.append_message(
            chat_state.make_assistant_message(
                f"⚠️ Could not reach the retrieval pipeline. Please retry once the backend is ready. ({exc})",
                source_ids=[],
            )
        )
        return

    if dispatch.chat.error and not dispatch.chat.answer:
        chat_state.append_message(
            chat_state.make_assistant_message(
                f"⚠️ {dispatch.chat.error}",
                source_ids=[],
            )
        )
    else:
        chat_state.append_message(
            chat_state.make_assistant_message(
                dispatch.chat.answer or "(no answer returned)",
                source_ids=[p.record_id for p in dispatch.chat.parent_contexts],
            )
        )

    chat_state.set_last_dispatch(dispatch)
    # Sources are no longer auto-opened — the user reveals them via the
    # "Sources" button rendered under the assistant message.

    msgs = chat_state.messages()
    if msgs:
        first_user = next((m for m in msgs if m.role == "user"), None)
        if first_user and chat_state.current_chat_id() is None:
            preview = first_user.content[:60]
            title = preview if len(preview) < 36 else preview[:34] + "..."
            chat_state.add_history_entry(
                title=title,
                preview=preview,
                message_count=len(msgs),
            )


def _render_chat_pane(sources: List[SourceViewModel]) -> None:
    messages = chat_state.messages()
    if not messages:
        empty_state.render(on_pick=_handle_submit)
    else:
        chat_messages.render(messages, sources)


def _render_map_pane(is_dark: bool) -> None:
    dispatch = chat_state.last_dispatch()
    if dispatch is None:
        st.info("Submit a question to populate the map, or explore baseline coverage below.")
        try:
            records, context_dict = baseline_map_payload()
        except Exception as exc:
            st.error(f"Map failed to load: {exc}")
            return
        map_view.render(records, context_dict, is_dark=is_dark)
        return

    # Show only the companies (and therefore counties) the answer actually cited.
    cited = extract_cited_company_names(dispatch.chat.parent_contexts)
    records = filter_records_to_companies(dispatch.map.records, cited)
    map_view.render(records, dispatch.map.context.to_dict(), is_dark=is_dark)


def _handle_submit(query: str) -> None:
    _process_query(query)
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Georgia EV Supply Chain Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _bootstrap_state()
    s = settings_state.settings()
    inject_styles(is_dark=s.is_dark_mode, compact=s.compact_mode)

    sidebar.render()
    settings_panel.render()

    header.render()
    # dashboard.render()  # Stat-card widget row commented out per request (#3).

    sources = _enrich_sources()
    mode = ui_state.view_mode()
    show_sources = ui_state.sources_panel_open() and bool(sources)

    if mode == "map":
        _render_map_pane(is_dark=s.is_dark_mode)
    elif mode == "split":
        if show_sources:
            chat_col, map_col, src_col = st.columns([0.34, 0.44, 0.22])
        else:
            chat_col, map_col = st.columns([0.5, 0.5])
            src_col = None
        with chat_col:
            _render_chat_pane(sources)
        with map_col:
            _render_map_pane(is_dark=s.is_dark_mode)
        if src_col is not None:
            with src_col:
                sources_panel.render(sources, s)
    else:
        if show_sources:
            chat_col, src_col = st.columns([0.7, 0.3])
        else:
            chat_col = st.container()
            src_col = None
        with chat_col:
            _render_chat_pane(sources)
        if src_col is not None:
            with src_col:
                sources_panel.render(sources, s)

    # st.chat_input MUST be at the top level of the script — it docks itself to
    # the bottom of the viewport. Skipped in the pure map view so it doesn't
    # cover the legend.
    if mode != "map":
        submitted = st.chat_input("Ask a question about Georgia's EV ecosystem...")
        if submitted:
            _handle_submit(submitted)


main()
