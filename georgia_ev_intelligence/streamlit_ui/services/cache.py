"""Cached factories + cached dispatcher entry point.

We centralize @st.cache_resource and @st.cache_data here so individual
services and components do not need to know about Streamlit's caching API.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.factory import (
    build_default_pipeline,
)

from ..bootstrap.build_companies_db import (
    DEFAULT_DB_PATH,
    DEFAULT_EXCEL_PATH,
    DEFAULT_GEOJSON_PATH,
    ensure_companies_db,
)
from ..spatial.query_planner import QueryPlanner
from ..spatial.spatial_engine import SpatialEngine
from .chat_service import ChatService
from .interfaces import DispatchResult
from .map_service import MapService
from .query_dispatcher import QueryDispatcher


@st.cache_resource(show_spinner="Loading hybrid retrieval pipeline...")
def get_chat_service() -> ChatService:
    return ChatService(retrieval_pipeline_factory=build_default_pipeline)


@st.cache_resource(show_spinner="Loading spatial engine...")
def get_map_service() -> MapService:
    db_path = ensure_companies_db()
    engine = SpatialEngine(db_path=db_path, geojson_path=DEFAULT_GEOJSON_PATH)
    planner = QueryPlanner(
        company_names=engine.list_company_names(),
        county_names=engine.county_names,
    )
    return MapService(spatial_engine=engine, query_planner=planner)


@st.cache_resource
def get_query_dispatcher() -> QueryDispatcher:
    return QueryDispatcher(chat_service=get_chat_service(), map_service=get_map_service())


@st.cache_resource(show_spinner="Indexing source workbook...")
def get_xlsx_lookup() -> Dict[int, Dict[str, Any]]:
    """Map 0-based source_row_id → row dict for source enrichment.

    Keys match the source_row_id values produced by
    offline_pipeline/chunking/parent_chunk.py (0..N-1).
    """
    if not DEFAULT_EXCEL_PATH.exists():
        return {}
    df = pd.read_excel(DEFAULT_EXCEL_PATH, sheet_name=0)
    df.columns = [str(c).strip().lower().replace("/", "_").replace(" ", "_") for c in df.columns]
    rename = {
        "company": "company",
        "category": "category",
        "industry_group": "industry_group",
        "updated_location": "location",
        "address": "address",
        "latitude": "latitude",
        "longitude": "longitude",
        "ev_supply_chain_role": "ev_supply_chain_role",
        "primary_oems": "primary_oems",
        "product___service": "product_service",
        "ev___battery_relevant": "ev_battery_relevant",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    lookup: Dict[int, Dict[str, Any]] = {}
    for row_index, row in df.iterrows():
        lookup[int(row_index)] = {
            "company": row.get("company"),
            "category": row.get("category"),
            "location": row.get("location"),
            "city": _city_from_location(row.get("location")),
            "county": _county_from_location(row.get("location")),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "ev_supply_chain_role": row.get("ev_supply_chain_role"),
            "product_service": row.get("product_service"),
        }
    return lookup


def _city_from_location(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text.split(",")[0].strip()


def _county_from_location(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    parts = [p.strip() for p in text.split(",") if p.strip()]
    for part in parts[1:]:
        if "county" in part.lower():
            return part.replace("County", "").replace("county", "").strip()
    return ""


@st.cache_resource
def get_county_geojson() -> dict:
    if not DEFAULT_GEOJSON_PATH.exists():
        return {}
    with DEFAULT_GEOJSON_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False, ttl=1800)
def dispatch_query_cached(query: str) -> DispatchResult:
    """Run dispatch once per unique query (30-minute TTL)."""
    dispatcher = get_query_dispatcher()
    return dispatcher.dispatch(query)


@st.cache_data(show_spinner=False, ttl=3600)
def baseline_map_payload() -> tuple:
    """Map records + context for the baseline (no-query) view, cached for 1 hour."""
    result = get_map_service().locate("")
    return list(result.records), result.context.to_dict()


@lru_cache(maxsize=1)
def project_data_dir() -> Path:
    return DEFAULT_GEOJSON_PATH.parent
