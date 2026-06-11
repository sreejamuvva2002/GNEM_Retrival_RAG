"""Expandable answer provenance rendered below the latest assistant answer.

Header ("Sources" / "{n} sources found"), then one expandable row per source.
Expanding a row reveals a two-column field grid
(Record ID, Category, Industry Group, Location, Address, Latitude, Longitude,
Facility Type, EV Supply Chain Role, Primary OEMs, Supplier Type, Employment,
Product/Service, EV/Battery Relevant) — matching the React grid.
"""
from __future__ import annotations

import html
from typing import Optional

import streamlit as st

from ..models.source import Provenance, SourceViewModel


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


def _subtitle(provenance: Provenance) -> str:
    if provenance.kind == "count":
        return "Derived from a count query"
    if provenance.kind == "groups":
        n = len(provenance.group_rows)
        return f"{n} group{'' if n == 1 else 's'} found"
    n = len(provenance.sources)
    return f"{n} source{'' if n == 1 else 's'} found"


def _render_group_rows(provenance: Provenance) -> None:
    """Aggregate/count answers have no per-company record; show the rows behind
    the number as their provenance."""
    if not provenance.group_rows:
        return
    st.caption("Result rows")
    st.dataframe(provenance.group_rows, use_container_width=True, hide_index=True)


def render(provenance: Provenance) -> None:
    st.markdown(
        f"""
        <div class="sources-header" id="sources-inline-anchor">
            <div>
                <div class="sources-header__title">Sources</div>
                <div class="sources-header__subtitle">
                    {_subtitle(provenance)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not provenance.has_content():
        st.info("Submit a question to see retrieved sources here.")
        return

    # Keep long provenance compact inside the scrollable conversation.
    with st.container(height=280):
        if provenance.sources:
            for source in provenance.sources:
                name = source.title or source.record_id or "Source"
                with st.expander(name):
                    st.markdown(_grid_html(source), unsafe_allow_html=True)
        else:
            _render_group_rows(provenance)
