"""Mock-seeded Streamlit harness that renders one UI state per page load.

Run directly with Streamlit and select the state via the ``?state=`` query param:

    streamlit run georgia_ev_intelligence/streamlit_ui/capture/capture_app.py \
        --server.port 8502 --server.headless true

States: empty | chat | loading | sources | settings | sidebar

This mirrors ``app.py``'s layout (same columns / header / panes / divider /
styles) but feeds hand-built fixtures so every backend-dependent state (chat,
loading, sources) renders WITHOUT a live RAG backend or database. Used only by
``shoot.py`` to capture screenshots for Figma — not part of the production app.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import streamlit as st

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from georgia_ev_intelligence.streamlit_ui.components import (
    chat_messages,
    empty_state,
    header,
    loading_card,
    map_view,
    resizable_split,
    settings_panel,
    sidebar,
    sources_panel,
)
from georgia_ev_intelligence.streamlit_ui.models.source import SourceViewModel
from georgia_ev_intelligence.streamlit_ui.state import chat_state, settings_state, ui_state
from georgia_ev_intelligence.streamlit_ui.theming.styles import inject_styles


# --------------------------------------------------------------------------- #
# Fixtures — realistic Georgia EV supply-chain sources (no DB lookup needed).
# --------------------------------------------------------------------------- #
def _mock_sources() -> List[SourceViewModel]:
    raw = [
        dict(
            title="Hyundai Motor Group Metaplant America",
            source_type="company",
            location_name="Ellabell",
            lat=32.157, lon=-81.420,
            category="OEM",
            industry_group="Automotive Manufacturing",
            location="Bryan County, GA",
            address="Highway 280, Ellabell, GA 31308",
            facility_type="EV Assembly Plant",
            ev_role="Final vehicle assembly",
            primary_oems="Hyundai, Kia, Genesis",
            supplier_type=None,
            employment="8500",
            product="Electric vehicle assembly (IONIQ line)",
            ev_relevant="Yes",
            snippet="Hyundai's $7.6B Metaplant in Bryan County is the OEM anchor of "
                    "Georgia's EV cluster, producing IONIQ-series electric vehicles.",
        ),
        dict(
            title="SK Battery America",
            source_type="supply_chain",
            location_name="Commerce",
            lat=34.204, lon=-83.457,
            category="Tier 1",
            industry_group="Battery Manufacturing",
            location="Jackson County, GA",
            address="One SK Blvd, Commerce, GA 30529",
            facility_type="Battery Cell Plant",
            ev_role="EV battery cell production",
            primary_oems="Ford, Volkswagen",
            supplier_type="Cell manufacturer",
            employment="3000",
            product="Lithium-ion EV battery cells",
            ev_relevant="Yes",
            snippet="SK Battery America operates two plants in Commerce supplying "
                    "lithium-ion cells to Ford and Volkswagen EV programs.",
        ),
        dict(
            title="Rivian Automotive — Stanton Springs",
            source_type="company",
            location_name="Social Circle",
            lat=33.616, lon=-83.711,
            category="OEM",
            industry_group="Automotive Manufacturing",
            location="Morgan/Walton County, GA",
            address="Stanton Springs North, Social Circle, GA 30025",
            facility_type="EV Assembly Plant (planned)",
            ev_role="Final vehicle assembly",
            primary_oems="Rivian",
            supplier_type=None,
            employment="7500",
            product="R2 / R3 electric SUVs and trucks",
            ev_relevant="Yes",
            snippet="Rivian's planned East Coast plant at Stanton Springs will build "
                    "the R2 platform, adding a second OEM anchor to the state.",
        ),
        dict(
            title="Qcells (Hanwha Solutions)",
            source_type="manufacturing",
            location_name="Dalton",
            lat=34.769, lon=-84.970,
            category="Tier 2",
            industry_group="Clean Energy Components",
            location="Whitfield County, GA",
            address="1400 Qcells Way, Dalton, GA 30721",
            facility_type="Solar Module Plant",
            ev_role="Charging-infrastructure energy supply",
            primary_oems=None,
            supplier_type="Component manufacturer",
            employment="2400",
            product="Solar PV modules for EV charging energy",
            ev_relevant="Indirect",
            snippet="Qcells' Dalton solar manufacturing underpins clean-energy supply "
                    "for Georgia's growing EV charging footprint.",
        ),
        dict(
            title="Georgia Department of Economic Development",
            source_type="government",
            location_name="Atlanta",
            lat=33.749, lon=-84.388,
            category="Government",
            industry_group="Public Sector",
            location="Fulton County, GA",
            address="75 Fifth Street NW, Atlanta, GA 30308",
            facility_type="State Agency",
            ev_role="EV cluster recruitment & incentives",
            primary_oems=None,
            supplier_type=None,
            employment=None,
            product="Economic development & EV incentive programs",
            ev_relevant="Policy",
            snippet="GDEcD coordinates incentives and workforce programs that recruited "
                    "Hyundai, Rivian and SK to Georgia.",
        ),
        dict(
            title="EV Charging Market Outlook (web)",
            source_type="web",
            location_name=None,
            lat=None, lon=None,
            category=None,
            industry_group=None,
            location=None,
            address=None,
            facility_type=None,
            ev_role=None,
            primary_oems=None,
            supplier_type=None,
            employment=None,
            product=None,
            ev_relevant=None,
            snippet="Industry coverage on the expansion of DC fast-charging corridors "
                    "across the I-85 and I-75 corridors in Georgia.",
        ),
    ]
    total = len(raw)
    sources: List[SourceViewModel] = []
    for i, r in enumerate(raw):
        rank = i + 1
        sources.append(
            SourceViewModel(
                id=f"KB_ROW_{i}",
                title=r["title"],
                snippet=r["snippet"],
                source_type=r["source_type"],
                location_name=r["location_name"],
                rank=rank,
                rank_score=max(0.0, 1.0 - (rank - 1) / total),
                record_id=f"KB_ROW_{i}" if r["source_type"] != "web" else f"WEB_{i}",
                source_row_id=i,
                parent_chunk_text=r["snippet"],
                latitude=r["lat"],
                longitude=r["lon"],
                category=r["category"],
                industry_group=r["industry_group"],
                location=r["location"],
                address=r["address"],
                facility_type=r["facility_type"],
                ev_supply_chain_role=r["ev_role"],
                primary_oems=r["primary_oems"],
                supplier_type=r["supplier_type"],
                employment=r["employment"],
                product_service=r["product"],
                ev_battery_relevant=r["ev_relevant"],
            )
        )
    return sources


_USER_QUERY = "Which battery manufacturers and EV plants operate in Georgia?"

_ASSISTANT_ANSWER = (
    "Georgia hosts a fast-growing EV manufacturing cluster anchored by two OEM "
    "assembly plants and several battery suppliers:\n\n"
    "- **Hyundai Metaplant America** in Bryan County builds IONIQ-series electric "
    "vehicles and is the largest economic-development project in state history [1].\n"
    "- **SK Battery America** in Commerce supplies lithium-ion cells to Ford and "
    "Volkswagen EV programs from two plants [2].\n"
    "- **Rivian** is developing an assembly plant at Stanton Springs for its R2 "
    "platform, adding a second OEM anchor [3].\n\n"
    "Supporting suppliers such as **Qcells** in Dalton extend the clean-energy "
    "base for charging infrastructure [4]."
)


def _seed_conversation() -> None:
    chat_state.append_message(chat_state.make_user_message(_USER_QUERY))
    chat_state.append_message(
        chat_state.make_assistant_message(
            _ASSISTANT_ANSWER,
            source_ids=[f"KB_ROW_{i}" for i in range(4)],
        )
    )


def _seed_history() -> None:
    samples = [
        ("Which battery manufacturers operate in Georgia?", 4),
        ("Show suppliers supporting Hyundai's EV ecosystem", 6),
        ("What charging infrastructure companies are active?", 2),
        ("Tier 1 automotive suppliers near Savannah", 8),
        ("Rivian plant timeline and workforce", 3),
    ]
    for title, count in samples:
        chat_state.add_history_entry(title=title, preview=title, message_count=count)
    # Leave no chat "active" so the first entry isn't force-highlighted oddly.
    chat_state.set_current_chat_id(None)


# --------------------------------------------------------------------------- #
# Panes (mirror app.py)
# --------------------------------------------------------------------------- #
def _render_map(sources: List[SourceViewModel], *, empty: bool) -> None:
    records = [] if empty else sources_panel.map_records(sources)
    map_view.render(records, {}, is_dark=False, height=600)


def _show_sidebar_css() -> None:
    """Undo the production rule that hides the sidebar, for the sidebar state."""
    st.markdown(
        "<style>[data-testid='stSidebar']{display:flex !important;}</style>",
        unsafe_allow_html=True,
    )


def main() -> None:
    state = "empty"
    try:
        state = (st.query_params.get("state") or "empty").lower()
    except Exception:
        state = "empty"

    st.set_page_config(
        page_title="Georgia EV — UI capture",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded" if state == "sidebar" else "collapsed",
    )

    chat_state.initialize()
    ui_state.initialize()
    settings_state.initialize()
    inject_styles(is_dark=False, compact=False)

    sources = _mock_sources()
    s = settings_state.settings()

    if state == "sidebar":
        _show_sidebar_css()
        _seed_history()
        sidebar.render()

    # Seed message/panel state per screen.
    show_sources = False
    if state in {"chat", "sources", "settings"}:
        _seed_conversation()
    if state == "sources":
        ui_state.set_sources_panel_open(True)
        show_sources = True

    messages = chat_state.messages()

    chat_col, map_col = st.columns([0.5, 0.5])

    loading_ph = None
    with chat_col:
        resizable_split.anchor()
        header.render()
        with st.container(key="chat_scroll"):
            if state == "loading":
                # Just the user bubble, then the animated loading card under it.
                chat_messages.render([chat_state.make_user_message(_USER_QUERY)], [])
                loading_ph = st.empty()
            elif messages:
                chat_messages.render(messages, sources)
            else:
                empty_state.render()
        st.chat_input("Ask about EV companies in Georgia...", key="chat_input")

    with map_col:
        _render_map(sources, empty=state in {"empty", "loading"})
        if show_sources:
            with st.container(key="gnem_sources"):
                sources_panel.render(sources, s)

    resizable_split.render()

    if loading_ph is not None:
        loading_card.render_step(loading_ph, active_index=2, completed_count=2)

    if state == "settings":
        settings_panel.open_settings()


main()
