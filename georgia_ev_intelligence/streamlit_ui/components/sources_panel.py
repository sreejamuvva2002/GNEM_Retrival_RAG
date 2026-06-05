"""Right-side sources panel — replica of chat-interface-with-map/sources-panel.tsx.

Header ("Sources" / "{n} sources found"), a "Close panel" button, then one
expandable row per source. Expanding a row reveals a two-column field grid
(Record ID, Category, Industry Group, Location, Address, Latitude, Longitude,
Facility Type, EV Supply Chain Role, Primary OEMs, Supplier Type, Employment,
Product/Service, EV/Battery Relevant) — matching the React grid.
"""
from __future__ import annotations

import html
import json
import math
from typing import List, Optional

import streamlit as st
import streamlit.components.v1 as components

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

    # The scrollable list is a single custom component: a 2-per-line grid with a
    # single-selection accordion, the no-location icon, a bottom scroll fade, and
    # client-side map focus. The component iframe is height-sized to fill the
    # panel by resizable_split.layoutHeights() (targets `.st-key-gnem_sources iframe`).
    _render_source_grid(sources)


def _source_coords(source: SourceViewModel) -> Optional[List[float]]:
    """Return [lat, lon] for a source, or None when the location is unavailable.

    Web/news sources and KB rows without workbook coordinates have no location;
    we also reject NaN/non-numeric values so the map is never asked to fly to an
    invalid point.
    """
    try:
        lat = float(source.latitude)  # type: ignore[arg-type]
        lon = float(source.longitude)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    return [lat, lon]


def map_records(sources: List[SourceViewModel]) -> List[dict]:
    """Build map marker records straight from the sources, de-duped by location.

    Guarantees source↔pin parity: a source yields a marker IFF it has a usable
    location (same `_source_coords` check that decides the no-location icon), so
    every located source is clickable to a real pin. Field names match what
    map_view._build_map consumes (company/latitude/longitude/address/product_service).
    """
    seen: set = set()
    records: List[dict] = []
    for source in sources:
        coords = _source_coords(source)
        if coords is None:
            continue
        key = (round(coords[0], 5), round(coords[1], 5))
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "company": source.title,
                "latitude": coords[0],
                "longitude": coords[1],
                "address": source.address,
                "product_service": source.product_service,
                "map_weight": 0.6,
            }
        )
    return records


# lucide "map-pin-off" — shown on sources whose location is unavailable.
_NO_LOCATION_SVG = (
    "<svg width='13' height='13' viewBox='0 0 24 24' fill='none' "
    "stroke='currentColor' stroke-width='2' stroke-linecap='round' "
    "stroke-linejoin='round'>"
    "<path d='M5.43 5.43A8.06 8.06 0 0 0 4 10c0 6 8 12 8 12a29.94 29.94 0 0 0 5-5'/>"
    "<path d='M19.18 13.52A8.66 8.66 0 0 0 20 10a8 8 0 0 0-8-8 7.88 7.88 0 0 0-3.52.82'/>"
    "<path d='M9.13 9.13A2.78 2.78 0 0 0 9 10a3 3 0 0 0 3 3 2.78 2.78 0 0 0 .87-.13'/>"
    "<line x1='2' y1='2' x2='22' y2='22'/></svg>"
)
# lucide "chevron-right" — the per-card expand caret.
_CHEVRON_SVG = (
    "<svg width='15' height='15' viewBox='0 0 24 24' fill='none' "
    "stroke='currentColor' stroke-width='2' stroke-linecap='round' "
    "stroke-linejoin='round'><polyline points='9 18 15 12 9 6'/></svg>"
)


def _render_source_grid(sources: List[SourceViewModel]) -> None:
    """Render the sources as a self-contained 2-column accordion component.

    All interaction (expand/collapse, highlight, map focus) is client-side, so
    there is no rerun/iframe-reload on click. Located cards call the map iframe's
    window.__gnemFocus(lat, lon); location-less cards show the map-pin-off icon
    and never move the map.
    """
    cards = []
    for i, source in enumerate(sources):
        coords = _source_coords(source)
        cards.append(
            {
                "idx": i,
                "title": source.title or source.record_id or "Source",
                "grid": _grid_html(source),
                "lat": None if coords is None else coords[0],
                "lon": None if coords is None else coords[1],
            }
        )

    payload = json.dumps(cards)
    components.html(
        f"""
        <!DOCTYPE html><html><head><meta charset="utf-8"><style>
          :root {{
            --fg:#1e293b; --muted:#64748b; --border:#e2e8f0; --bg:#ffffff;
            --primary:#0f766e; --sel-bg:#f0fdfa; --sel-border:#5eead4;
          }}
          * {{ box-sizing: border-box; }}
          html, body {{ margin:0; height:100%; }}
          body {{ font-family:'Inter',system-ui,-apple-system,sans-serif;
                  color:var(--fg); background:transparent; }}
          /* Scroll container fills the iframe; bottom padding leaves room for the fade. */
          #scroll {{ height:100%; overflow-y:auto; overflow-x:hidden;
                     padding:2px 2px 26px 2px; }}
          #grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px;
                   align-items:start; }}
          @media (max-width:430px) {{ #grid {{ grid-template-columns:1fr; }} }}
          .card {{ border:1px solid var(--border); border-radius:12px;
                   background:#f8fafc; overflow:hidden; align-self:start;
                   transition:border-color .15s ease, box-shadow .15s ease,
                              background .15s ease; }}
          .card:hover {{ border-color:#cbd5e1; }}
          .card.sel {{ border-color:var(--sel-border); background:var(--sel-bg);
                       box-shadow:0 0 0 1px var(--sel-border); }}
          .head {{ display:flex; align-items:center; gap:6px; cursor:pointer;
                   padding:10px 11px; font-weight:600; font-size:13px;
                   user-select:none; }}
          .chev {{ color:var(--muted); flex:0 0 auto; display:inline-flex;
                   transition:transform .18s ease; }}
          .card.open .chev {{ transform:rotate(90deg); }}
          .title {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
          .noloc {{ margin-left:auto; color:#94a3b8; flex:0 0 auto;
                    display:inline-flex; align-items:center; }}
          .detail {{ display:none; padding:0 11px 11px 11px; }}
          .card.open .detail {{ display:block; }}
          .source-grid {{ display:grid; grid-template-columns:1fr 1fr;
                          gap:8px 14px; font-size:12px; }}
          .source-field {{ display:flex; flex-direction:column; min-width:0; }}
          .source-field--full {{ grid-column:span 2; }}
          .source-field__label {{ color:var(--muted); font-size:11px; }}
          .source-field__value {{ color:var(--fg); overflow-wrap:anywhere; }}
          /* Bottom fade — scroll affordance. */
          #fade {{ position:absolute; left:0; right:0; bottom:0; height:30px;
                   pointer-events:none; background:linear-gradient(to bottom,
                   rgba(255,255,255,0), rgba(255,255,255,0.96)); transition:opacity .2s; }}
        </style></head><body>
          <div id="scroll"><div id="grid"></div></div>
          <div id="fade"></div>
          <script>
            const CARDS = {payload};
            const CHEV = `{_CHEVRON_SVG}`;
            const NOLOC = `{_NO_LOCATION_SVG}`;
            const grid = document.getElementById('grid');
            const scroll = document.getElementById('scroll');
            const fade = document.getElementById('fade');
            let selected = null;

            function focusMap(lat, lon) {{
              try {{
                const mi = window.parent.document.querySelector('.st-key-gnem_map iframe');
                if (mi && mi.contentWindow && mi.contentWindow.__gnemFocus) {{
                  mi.contentWindow.__gnemFocus(lat, lon);
                }}
              }} catch (e) {{}}
            }}

            CARDS.forEach(function(c) {{
              const card = document.createElement('div');
              card.className = 'card';
              const has = c.lat != null && c.lon != null;
              const head = document.createElement('div');
              head.className = 'head';
              head.innerHTML = '<span class="chev">' + CHEV + '</span>'
                + '<span class="title">' + (c.title || 'Source') + '</span>'
                + (has ? '' : '<span class="noloc" title="Location not available" '
                    + 'aria-label="Location not available">' + NOLOC + '</span>');
              const detail = document.createElement('div');
              detail.className = 'detail';
              detail.innerHTML = c.grid;
              card.appendChild(head);
              card.appendChild(detail);
              head.addEventListener('click', function() {{
                if (selected === card) {{           // toggle the open card closed
                  card.classList.remove('open', 'sel');
                  selected = null;
                  return;
                }}
                if (selected) selected.classList.remove('open', 'sel');
                card.classList.add('open', 'sel');  // single-selection accordion
                selected = card;
                if (has) focusMap(c.lat, c.lon);    // located → recenter map
              }});
              grid.appendChild(card);
            }});

            function updateFade() {{
              const atBottom = scroll.scrollTop + scroll.clientHeight
                  >= scroll.scrollHeight - 2;
              const scrollable = scroll.scrollHeight > scroll.clientHeight + 2;
              fade.style.opacity = (scrollable && !atBottom) ? '1' : '0';
            }}
            scroll.addEventListener('scroll', updateFade);
            window.addEventListener('resize', updateFade);
            setTimeout(updateFade, 60);
          </script>
        </body></html>
        """,
        height=400,
    )
