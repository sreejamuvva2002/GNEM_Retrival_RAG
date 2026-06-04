"""Right-side sources panel — replica of chat-interface-with-map/sources-panel.tsx.

Header ("Sources" / "{n} sources found"), a "Close panel" button, then one
expandable row per source. Expanding a row reveals a two-column field grid
(Record ID, Category, Industry Group, Location, Address, Latitude, Longitude,
Facility Type, EV Supply Chain Role, Primary OEMs, Supplier Type, Employment,
Product/Service, EV/Battery Relevant) — matching the React grid.
"""
from __future__ import annotations

import html
from typing import List, Optional

import streamlit as st

from ..models.chat import Settings
from ..models.source import SourceViewModel
from ..state import ui_state


def _field(label: str, value: Optional[str], *, full: bool = False) -> str:
    if value is None or not str(value).strip():
        return ""
    cls = "source-field source-field--full" if full else "source-field"
    return (
        f"<div class='{cls}'>"
        f"<span class='source-field__label'>{html.escape(label)}</span>"
        f"<span class='source-field__value'>{html.escape(str(value))}</span>"
        "</div>"
    )


def _grid_html(source: SourceViewModel) -> str:
    lat = "" if source.latitude is None else f"{source.latitude}"
    lon = "" if source.longitude is None else f"{source.longitude}"
    fields = [
        _field("Record ID", source.record_id),
        _field("Category", source.category),
        _field("Industry Group", source.industry_group),
        _field("Location", source.location or source.location_name),
        _field("Address", source.address, full=True),
        _field("Latitude", lat),
        _field("Longitude", lon),
        _field("Facility Type", source.facility_type),
        _field("EV Supply Chain Role", source.ev_supply_chain_role),
        _field("Primary OEMs", source.primary_oems),
        _field("Supplier Type", source.supplier_type),
        _field("Employment", source.employment),
        _field("Product/Service", source.product_service, full=True),
        _field("EV/Battery Relevant", source.ev_battery_relevant),
    ]
    return f"<div class='source-grid'>{''.join(fields)}</div>"


def render(sources: List[SourceViewModel], settings: Settings) -> None:
    st.markdown(
        f"""
        <div class="sources-header">
            <div>
                <div class="sources-header__title">Sources</div>
                <div class="sources-header__subtitle">
                    {len(sources)} source{'s' if len(sources) != 1 else ''} found
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("✕ Close panel", key="sources_close", use_container_width=True):
        ui_state.set_sources_panel_open(False)
        st.rerun()

    if not sources:
        st.info("Submit a question to see retrieved sources here.")
        return

    # Native fixed-height container = scrollable list, kept tight so the whole
    # right column (shrunk map + this panel + docked chat input) fits in one
    # viewport without page scroll.
    with st.container(height=280):
        for source in sources:
            name = source.title or source.record_id or "Source"
            with st.expander(name):
                st.markdown(_grid_html(source), unsafe_allow_html=True)
