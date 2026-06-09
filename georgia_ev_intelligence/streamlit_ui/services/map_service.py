"""Map service: derive a MapResult from the user's question + spatial engine."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..models.map import MapContext, MapResult
from ..spatial.postgis_spatial_engine import PostGISSpatialEngine
from ..spatial.query_planner import QueryPlanner
from .interfaces import IMapDataService


class MapService(IMapDataService):
    """Run the spatial pipeline (QueryPlanner → SpatialEngine) for a question."""

    def __init__(self, spatial_engine: PostGISSpatialEngine, query_planner: QueryPlanner) -> None:
        self._spatial_engine = spatial_engine
        self._query_planner = query_planner

    def locate(self, query: str) -> MapResult:
        plan = self._query_planner.plan((query or "").strip())
        hints: Dict[str, Any] = plan.get("hints", {}) or {}
        analysis_intent = str(hints.get("analysis_intent") or "standard_retrieval")

        records_df = pd.DataFrame()
        context = MapContext(map_mode="standard")
        if plan.get("classification") == "NO_RETRIEVAL":
            context.map_mode = "no_retrieval"
            return MapResult(records=[], context=context)

        coords_hint = hints.get("coordinates") or {}
        center_lat = _safe_float(coords_hint.get("lat"))
        center_lon = _safe_float(coords_hint.get("lon"))
        radius_km = _safe_float(hints.get("radius_km"))
        counties = list(hints.get("counties") or [])
        city = hints.get("city") or hints.get("facility_city")
        company_name = hints.get("company_name")

        # Resolve a city/company to coordinates if no explicit lat/lon was given.
        if (center_lat is None or center_lon is None) and city:
            resolved = self._spatial_engine.resolve_place_coordinates(str(city))
            if resolved:
                center_lat, center_lon = resolved
                context.focus_label = str(city).title()

        if (center_lat is None or center_lon is None) and company_name:
            resolved = self._spatial_engine.resolve_place_coordinates(str(company_name))
            if resolved:
                center_lat, center_lon = resolved
                context.focus_label = str(company_name)

        # Branch on intent / hints.
        if analysis_intent == "gap_analysis":
            context.map_mode = "gap_analysis"
            context.gap_report = self._spatial_engine.supply_gap_report(
                capability_term=hints.get("capability_term"),
                category_term=hints.get("category_term"),
                county_scope=counties or None,
            )
            records_df = self._spatial_engine.companies_df.copy()
        elif center_lat is not None and center_lon is not None and radius_km:
            context.map_mode = "radius_search"
            context.center_lat = center_lat
            context.center_lon = center_lon
            context.radius_km = radius_km
            records_df = self._spatial_engine.companies_within_radius(
                lat=center_lat, lon=center_lon, radius_km=radius_km
            )
        elif counties:
            context.map_mode = "county_filter"
            context.counties = counties
            records_df = self._spatial_engine.companies_in_counties(counties)
            resolved = self._spatial_engine.resolve_place_coordinates(str(counties[0]))
            if resolved:
                context.center_lat, context.center_lon = resolved
        else:
            # No spatial signal — render every company we know about.
            context.map_mode = "standard"
            records_df = self._spatial_engine.companies_df.copy()

        # Always supply a center point for map_view (fallback to records average).
        if context.center_lat is None and not records_df.empty:
            valid = records_df.dropna(subset=["latitude", "longitude"])
            if not valid.empty:
                context.center_lat = float(valid["latitude"].mean())
                context.center_lon = float(valid["longitude"].mean())

        # County coverage count for the dashboard widgets.
        if "county" in records_df.columns:
            unique_counties = (
                records_df["county"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().unique()
            )
            context.county_coverage_count = int(len(unique_counties))

        records = _records_to_list(records_df)
        return MapResult(records=records, context=context)


def filter_records_to_companies(
    records: List[Dict[str, Any]], normalized_names: set
) -> List[Dict[str, Any]]:
    """Keep only map records whose company matches one of the cited companies.

    `normalized_names` is a set of normalized company names (see
    chat_service.extract_cited_company_names). Matching is exact after
    normalization so similarly named companies do not create extra markers. If
    the cited set is empty, no company markers are shown.
    """
    if not normalized_names:
        return []

    from .chat_service import normalize_company_name

    kept: List[Dict[str, Any]] = []
    for record in records:
        haystack = normalize_company_name(str(record.get("company") or ""))
        if not haystack:
            continue
        if haystack in normalized_names:
            kept.append(record)
    return kept


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _records_to_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    df = df.copy()

    if "latitude" in df.columns and "longitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])

    # map_view expects a normalized map_weight in [0,1].
    if "distance_km" in df.columns and df["distance_km"].notna().any():
        max_distance = df["distance_km"].max() or 1.0
        df["map_weight"] = (1.0 - (df["distance_km"].fillna(max_distance) / max_distance)).clip(0.05, 1.0)
    else:
        df["map_weight"] = 0.6

    return df.to_dict(orient="records")
