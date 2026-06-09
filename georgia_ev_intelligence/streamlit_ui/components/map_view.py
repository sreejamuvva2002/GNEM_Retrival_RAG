"""Map view — folium/Leaflet replica of chat-interface-with-map/map-panel.tsx.

The React panel is a plain Leaflet map: OpenStreetMap tiles, default blue pin
markers, and a popup per company (bold name / address / product-service). We
reproduce that exactly with folium + streamlit-folium. The richer PyDeck
analytical layers (county choropleth, heatmap, radius, arcs) were intentionally
dropped to match the React look.
"""
from __future__ import annotations

import html
from typing import Any, Dict, List

import folium
import streamlit as st
import streamlit.components.v1 as components

# Georgia center + zoom, matching map-panel.tsx (georgiaCenter / zoom={7}).
GEORGIA_CENTER = [33.2, -84.3]
GEORGIA_ZOOM = 7

_EMPTY_OVERLAY = (
    "<div style=\"position:absolute; inset:0; display:flex; align-items:center;"
    " justify-content:center; pointer-events:none; z-index:500;\">"
    "<div style=\"background:rgba(248,250,252,0.82); color:#64748b;"
    " font:600 13px/1.4 Inter,system-ui,sans-serif; padding:10px 16px;"
    " border-radius:10px;\">Ask a question to see company locations on the map</div>"
    "</div>"
)


def _coord(value: Any):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _company_block(record: Dict[str, Any]) -> str:
    """One company's name + detail lines for a popup."""
    company = html.escape(str(record.get("company") or "Company"))
    lines = [f"<p style='font-weight:600; margin:0 0 2px 0;'>{company}</p>"]
    for key in ("address", "product_service"):
        val = record.get(key)
        if val is not None and str(val).strip() and str(val).strip().lower() != "nan":
            lines.append(
                f"<p style='color:#64748b; margin:0;'>{html.escape(str(val).strip())}</p>"
            )
    return "".join(lines)


def _popup_html(records: List[Dict[str, Any]]) -> str:
    """Popup for one location; lists every company sharing that point."""
    if len(records) > 1:
        header = (
            f"<p style='font-weight:700; margin:0 0 6px 0; color:#0f172a;'>"
            f"{len(records)} companies at this location</p>"
        )
        blocks = "<hr style='border:none; border-top:1px solid #e2e8f0; margin:6px 0;'>".join(
            _company_block(r) for r in records
        )
        body = header + blocks
    else:
        body = _company_block(records[0])
    return (
        "<div style='font:13px/1.4 Inter,system-ui,sans-serif; min-width:180px;'>"
        + body
        + "</div>"
    )


def _group_by_location(records: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    """Bucket records by exact coordinate so co-located companies share one pin.

    Several companies can occupy the same physical site (e.g. an OEM campus), so
    plotting one marker per record stacks them invisibly. Grouping by rounded
    lat/lon keeps the plain-marker look while making every company reachable
    through the shared pin's popup.
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for record in records:
        lat = _coord(record.get("latitude"))
        lon = _coord(record.get("longitude"))
        if lat is None or lon is None:
            continue
        groups.setdefault((round(lat, 6), round(lon, 6)), []).append(record)
    return groups


def _build_map(records: List[Dict[str, Any]]) -> folium.Map:
    fmap = folium.Map(
        location=GEORGIA_CENTER,
        zoom_start=GEORGIA_ZOOM,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    for (lat, lon), group in _group_by_location(records).items():
        if len(group) > 1:
            names = ", ".join(str(r.get("company") or "") for r in group)
            tooltip = f"{len(group)} companies: {names}"
        else:
            tooltip = str(group[0].get("company") or "")
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(_popup_html(group), max_width=260),
            tooltip=tooltip,
        ).add_to(fmap)

    if not records:
        # Centered overlay inside the same iframe (matches the React empty state).
        fmap.get_root().html.add_child(folium.Element(_EMPTY_OVERLAY))

    return fmap


def render(
    records: List[Dict[str, Any]],
    map_context: Dict[str, Any] | None = None,
    *,
    is_dark: bool = False,
    height: int = 600,
) -> None:
    """Render the Leaflet map filling the available height.

    We embed folium's raw HTML (map div is height:100%) in a static
    `components.html` iframe inside a keyed container, then CSS sizes the iframe
    (full viewport height by default, ~44vh when the sources panel is open — see
    theming/styles.py and app.py). Leaflet reads the container size at init and
    auto-resizes when the divider changes the column width. `height` is only the
    pre-CSS fallback. `map_context`/`is_dark` kept for call-site compatibility.
    """
    fmap = _build_map(records)
    html_doc = fmap.get_root().render()
    with st.container(key="gnem_map"):
        components.html(html_doc, height=height)
