"""Geospatial map — ported pydeck render from PAST_GEO_MAP_VIEW/frontend/app.py.

Layers in render order:
  * County choropleth (selected / gap / coverage / neutral fills)
  * Radius circle (when a radius_km is set)
  * Hub-to-supplier arcs (when there is a center point)
  * Heatmap
  * Scatterplot (one dot per company)
  * Center marker (search hub)
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List

import pandas as pd
import pydeck as pdk
import streamlit as st

from ..services.cache import get_county_geojson


COORDINATE_SOURCE_LABELS = {
    "coordinates_excel": "Coordinate Workbook",
    "source_excel": "Source Excel",
    "county_centroid": "GeoJSON County Centroid",
    "missing": "Missing",
    "unknown": "Unknown",
}

COORDINATE_SOURCE_COLORS = {
    "coordinates_excel": [18, 137, 127, 220],
    "source_excel": [56, 111, 164, 220],
    "county_centroid": [216, 144, 47, 215],
    "missing": [185, 75, 92, 190],
    "unknown": [88, 100, 113, 185],
}


def _theme_colors(is_dark: bool) -> Dict[str, Any]:
    """Marker / overlay / tooltip colors that adapt to the active theme."""
    if is_dark:
        return {
            "marker_line": [231, 238, 245, 200],
            "county_line": [231, 238, 245, 90],
            "center_fill": [231, 238, 245, 235],
            "center_line": [15, 23, 42, 220],
            "tooltip_bg": "#102433",
            "tooltip_fg": "#ffffff",
        }
    return {
        "marker_line": [17, 38, 58, 190],
        "county_line": [17, 38, 58, 120],
        "center_fill": [15, 23, 42, 235],
        "center_line": [255, 255, 255, 220],
        "tooltip_bg": "#ffffff",
        "tooltip_fg": "#102433",
    }


def _normalize_coordinate_source(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return COORDINATE_SOURCE_LABELS["unknown"]
    text = str(value or "unknown").strip()
    lower = text.lower()
    if lower.startswith("coordinates_excel"):
        return COORDINATE_SOURCE_LABELS["coordinates_excel"]
    return COORDINATE_SOURCE_LABELS.get(lower, text or COORDINATE_SOURCE_LABELS["unknown"])


def _build_radius_circle_geojson(center_lat: float, center_lon: float, radius_km: float, steps: int = 96) -> dict:
    earth_radius_km = 6371.0088
    lat_rad = math.radians(center_lat)
    lon_rad = math.radians(center_lon)
    angular_distance = float(radius_km) / earth_radius_km
    coords = []
    for step in range(steps + 1):
        bearing = 2.0 * math.pi * step / steps
        point_lat = math.asin(
            math.sin(lat_rad) * math.cos(angular_distance)
            + math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing)
        )
        point_lon = lon_rad + math.atan2(
            math.sin(bearing) * math.sin(angular_distance) * math.cos(lat_rad),
            math.cos(angular_distance) - math.sin(lat_rad) * math.sin(point_lat),
        )
        coords.append([math.degrees(point_lon), math.degrees(point_lat)])
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"fill_color": [18, 137, 127, 26]},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        }],
    }


@st.cache_data(show_spinner=False)
def _build_overlay_cached(
    counts_items: tuple, gap_names: frozenset, selected: frozenset
) -> Dict | None:
    """Deep-copy the base GeoJSON and paint per-county fills, cached by signature.

    The 18 MB base GeoJSON is only deep-copied + recolored once per distinct
    (counts, gap, selected) combination instead of on every rerun / tab switch.
    """
    base = get_county_geojson()
    if not base:
        return None

    counts = dict(counts_items)
    max_count = max(counts.values()) if counts else 1

    overlay = json.loads(json.dumps(base))
    for feature in overlay.get("features", []):
        props = feature.setdefault("properties", {})
        county_name = str(props.get("NAME10") or "").strip()
        county_key = county_name.lower()
        count = int(counts.get(county_name, 0))
        intensity = min(180, 35 + int(180 * count / max_count)) if count else 18

        if county_key in gap_names:
            fill = [185, 75, 92, 120]
        elif county_key in selected:
            fill = [18, 137, 127, 95]
        elif count:
            fill = [56, 111, 164, intensity]
        else:
            fill = [110, 126, 142, 10]

        props["fill_color"] = fill
        props["facility_count"] = count
    return overlay


def _build_county_overlay_geojson(df: pd.DataFrame, map_context: Dict) -> Dict | None:
    counts: Dict[str, int] = {}
    if "county" in df.columns:
        counts = (
            df["county"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().value_counts().to_dict()
        )

    gap_report = map_context.get("gap_report") or {}
    gap_names = frozenset(
        str(item.get("county") or "").strip().lower()
        for item in (gap_report.get("gap_counties") or [])
        if str(item.get("county") or "").strip()
    )
    selected = frozenset(
        str(c or "").strip().lower().replace(" county", "")
        for c in (map_context.get("counties") or [])
        if str(c or "").strip()
    )
    return _build_overlay_cached(tuple(sorted(counts.items())), gap_names, selected)


def _build_center_and_arc_frames(df: pd.DataFrame, map_context: Dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    center_lat = map_context.get("center_lat")
    center_lon = map_context.get("center_lon")
    if center_lat is None or center_lon is None:
        return pd.DataFrame(), pd.DataFrame()

    center_df = pd.DataFrame([{
        "latitude": float(center_lat),
        "longitude": float(center_lon),
        "label": str(map_context.get("focus_label") or "Search center"),
        "radius": 7000,
    }])
    arc_df = df.copy()
    arc_df["source_latitude"] = float(center_lat)
    arc_df["source_longitude"] = float(center_lon)
    # df.get("map_weight", 0.5) returns the scalar 0.5 (no .apply) when the
    # column is absent — build a real Series so .apply always works.
    weights = (
        arc_df["map_weight"]
        if "map_weight" in arc_df.columns
        else pd.Series(0.5, index=arc_df.index)
    )
    arc_df["arc_width"] = weights.apply(lambda v: 1 + int(float(v) * 4))
    return center_df, arc_df.head(60)


def _legend_html() -> str:
    return """
    <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin: 0.4rem 0 0.8rem 0;">
        <span style="display:inline-flex; align-items:center; gap:0.35rem; padding:0.3rem 0.6rem;
                     border-radius:999px; background: var(--glass-bg); border:1px solid var(--glass-border);
                     font-size:0.72rem; color:var(--muted-fg);">
            <span style="width:9px; height:9px; border-radius:999px; background:#12897f;"></span>
            Coordinate Workbook
        </span>
        <span style="display:inline-flex; align-items:center; gap:0.35rem; padding:0.3rem 0.6rem;
                     border-radius:999px; background: var(--glass-bg); border:1px solid var(--glass-border);
                     font-size:0.72rem; color:var(--muted-fg);">
            <span style="width:9px; height:9px; border-radius:999px; background:#386fa4;"></span>
            Source Excel
        </span>
        <span style="display:inline-flex; align-items:center; gap:0.35rem; padding:0.3rem 0.6rem;
                     border-radius:999px; background: var(--glass-bg); border:1px solid var(--glass-border);
                     font-size:0.72rem; color:var(--muted-fg);">
            <span style="width:9px; height:9px; border-radius:999px; background:#d8902f;"></span>
            County Centroid
        </span>
        <span style="display:inline-flex; align-items:center; gap:0.35rem; padding:0.3rem 0.6rem;
                     border-radius:999px; background: var(--glass-bg); border:1px solid var(--glass-border);
                     font-size:0.72rem; color:var(--muted-fg);">
            <span style="width:9px; height:9px; border-radius:999px; background:#b94b5c;"></span>
            Missing
        </span>
    </div>
    """


def _deck_signature(
    records: List[Dict[str, Any]], map_context: Dict[str, Any], is_dark: bool
) -> str:
    """Stable hash so a tab switch with identical inputs reuses the built deck."""
    key = json.dumps(
        {
            "companies": sorted(str(r.get("company")) for r in records),
            "n": len(records),
            "dark": is_dark,
            "mode": map_context.get("map_mode"),
            "center": (map_context.get("center_lat"), map_context.get("center_lon")),
            "radius": map_context.get("radius_km"),
            "counties": sorted(str(c) for c in (map_context.get("counties") or [])),
        },
        default=str,
        sort_keys=True,
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def render(records: List[Dict[str, Any]], map_context: Dict[str, Any], *, is_dark: bool) -> None:
    st.markdown(_legend_html(), unsafe_allow_html=True)

    if not records:
        st.info("No mapped companies for this query.")
        return

    # Reuse the previously built deck when nothing changed (instant tab switch).
    signature = _deck_signature(records, map_context, is_dark)
    cached = st.session_state.get("_map_deck_cache")
    if cached and cached[0] == signature:
        st.pydeck_chart(cached[1], use_container_width=True)
        return

    deck = _build_deck(records, map_context, is_dark)
    if deck is None:
        return
    st.session_state["_map_deck_cache"] = (signature, deck)
    st.pydeck_chart(deck, use_container_width=True)


def _build_deck(
    records: List[Dict[str, Any]], map_context: Dict[str, Any], is_dark: bool
):
    colors = _theme_colors(is_dark)

    df = pd.DataFrame(records).copy()
    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.info("Returned company rows do not contain latitude/longitude columns.")
        return None

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    if df.empty:
        st.info("No valid coordinates available.")
        return None

    if "map_weight" not in df.columns:
        df["map_weight"] = 0.6
    df["map_weight"] = pd.to_numeric(df["map_weight"], errors="coerce").fillna(0.5).clip(0.05, 1.0)
    df["radius"] = df["map_weight"].apply(lambda v: 2800.0 + float(v) * 22000.0)
    df["heat_weight"] = df["map_weight"].apply(lambda v: 8.0 + float(v) * 88.0)
    if "coordinate_source" not in df.columns:
        df["coordinate_source"] = "unknown"

    def _color(value: Any) -> List[int]:
        text = str(value or "").lower()
        if text.startswith("coordinates_excel"):
            return COORDINATE_SOURCE_COLORS["coordinates_excel"]
        return COORDINATE_SOURCE_COLORS.get(text, COORDINATE_SOURCE_COLORS["unknown"])

    df["fill_color"] = df["coordinate_source"].apply(_color)
    df["tooltip_company"] = df.get("company", pd.Series(index=df.index)).fillna("Unknown company")
    df["tooltip_role"] = df.get("ev_supply_chain_role", pd.Series(index=df.index)).fillna("—")
    df["tooltip_product"] = df.get("product_service", pd.Series(index=df.index)).fillna("—")
    df["tooltip_location"] = (
        df.get("city", pd.Series(index=df.index)).fillna("")
        + ", "
        + df.get("county", pd.Series(index=df.index)).fillna("")
    ).str.strip(", ")
    df["tooltip_coord_source"] = df["coordinate_source"].apply(_normalize_coordinate_source)
    df["tooltip_weight"] = df["map_weight"].map(lambda v: f"{float(v):.2f}")

    view_state = pdk.ViewState(
        latitude=float(map_context.get("center_lat") or df["latitude"].mean()),
        longitude=float(map_context.get("center_lon") or df["longitude"].mean()),
        zoom=7.0 if map_context.get("map_mode") == "radius_search" else (6.5 if len(df) > 8 else 7.6),
        pitch=0,
    )

    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position="[longitude, latitude]",
        get_weight="heat_weight",
        radius_pixels=42,
        intensity=0.82,
        threshold=0.06,
        opacity=0.42,
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[longitude, latitude]",
        get_fill_color="fill_color",
        get_line_color=colors["marker_line"],
        line_width_min_pixels=1,
        stroked=True,
        filled=True,
        pickable=True,
        get_radius="radius",
    )

    layers = [heatmap_layer, scatter_layer]

    overlay_geojson = _build_county_overlay_geojson(df, map_context)
    if overlay_geojson:
        layers.insert(0, pdk.Layer(
            "GeoJsonLayer",
            data=overlay_geojson,
            stroked=True,
            filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=colors["county_line"],
            line_width_min_pixels=2,
            pickable=False,
        ))

    if (
        map_context.get("center_lat") is not None
        and map_context.get("center_lon") is not None
        and map_context.get("radius_km") is not None
        and map_context.get("map_mode") == "radius_search"
    ):
        layers.append(pdk.Layer(
            "GeoJsonLayer",
            data=_build_radius_circle_geojson(
                float(map_context["center_lat"]),
                float(map_context["center_lon"]),
                float(map_context["radius_km"]),
            ),
            stroked=True,
            filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=[18, 137, 127, 180],
            line_width_min_pixels=2,
            pickable=False,
        ))

    center_df, arc_df = _build_center_and_arc_frames(df, map_context)
    if not arc_df.empty:
        layers.append(pdk.Layer(
            "ArcLayer",
            data=arc_df,
            get_source_position="[source_longitude, source_latitude]",
            get_target_position="[longitude, latitude]",
            get_source_color=[18, 137, 127, 155],
            get_target_color="fill_color",
            get_width="arc_width",
            pickable=False,
        ))
    if not center_df.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=center_df,
            get_position="[longitude, latitude]",
            get_fill_color=colors["center_fill"],
            get_line_color=colors["center_line"],
            stroked=True,
            filled=True,
            line_width_min_pixels=2,
            radius_min_pixels=7,
            get_radius="radius",
            pickable=True,
        ))

    tooltip = {
        "html": (
            "<div style='font-family:Inter, sans-serif; min-width:220px;'>"
            "<div style='font-size:14px; font-weight:800; margin-bottom:8px;'>{tooltip_company}</div>"
            f"<div style='font-size:12px; line-height:1.6; color:{colors['tooltip_fg']};'>"
            "<b>Role:</b> {tooltip_role}<br/>"
            "<b>Product:</b> {tooltip_product}<br/>"
            "<b>Location:</b> {tooltip_location}<br/>"
            "<b>Coordinate:</b> {tooltip_coord_source}<br/>"
            "<b>Map weight:</b> {tooltip_weight}"
            "</div></div>"
        ),
        "style": {
            "backgroundColor": colors["tooltip_bg"],
            "color": colors["tooltip_fg"],
            "borderRadius": "16px",
            "padding": "14px 16px",
            "border": "1px solid rgba(127,127,127,0.18)",
        },
    }

    return pdk.Deck(
        map_style="dark_no_labels" if is_dark else "light_no_labels",
        initial_view_state=view_state,
        tooltip=tooltip,
        layers=layers,
    )
