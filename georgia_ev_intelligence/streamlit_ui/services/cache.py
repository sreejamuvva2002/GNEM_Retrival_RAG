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

from georgia_ev_intelligence.route_generation.route_service import (
    build_default_route_service,
)

from ..bootstrap.build_companies_db import (
    DEFAULT_EXCEL_PATH,
    DEFAULT_GEOJSON_PATH,
)
from ..spatial.postgis_spatial_engine import PostGISSpatialEngine
from ..spatial.query_planner import QueryPlanner
from .interfaces import DispatchResult, IChatService
from .map_service import MapService
from .query_dispatcher import QueryDispatcher
from .route_chat_service import RouteChatService


@st.cache_resource(show_spinner="Loading routing pipeline...")
def get_chat_service() -> IChatService:
    """Route-aware chat: router + validator + safe per-route executor (+ Ollama)."""
    return RouteChatService(route_service_factory=build_default_route_service, use_llm=True)


@st.cache_resource(show_spinner="Loading spatial engine...")
def get_map_service() -> MapService:
    engine = PostGISSpatialEngine()
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
        "supplier_or_affiliation_type": "supplier_type",
        "employment": "employment",
        "primary_facility_type": "facility_type",
        "product___service": "product_service",
        "ev___battery_relevant": "ev_battery_relevant",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    lookup: Dict[int, Dict[str, Any]] = {}
    for row_index, row in df.iterrows():
        lookup[int(row_index)] = {
            "company": row.get("company"),
            "category": row.get("category"),
            "industry_group": row.get("industry_group"),
            "location": row.get("location"),
            "city": _city_from_location(row.get("location")),
            "county": _county_from_location(row.get("location")),
            "address": row.get("address"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "facility_type": row.get("facility_type"),
            "ev_supply_chain_role": row.get("ev_supply_chain_role"),
            "primary_oems": row.get("primary_oems"),
            "supplier_type": row.get("supplier_type"),
            "employment": row.get("employment"),
            "product_service": row.get("product_service"),
            "ev_battery_relevant": row.get("ev_battery_relevant"),
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


def dispatch_query_cached(
    query: str, history: tuple[tuple[str, str], ...] | None = None, _on_step=None
) -> DispatchResult:
    """Run dispatch for one query, emitting live step events via `_on_step`.

    NOT cached: `_on_step` writes to a Streamlit layout block (the loading-card
    placeholder) created by the caller, which is illegal inside an
    `@st.cache_resource`/`cache_data` function ("a streamlit element is called
    on some layout block created outside the function"). Caching would also skip
    the step animation on a hit. The heavy state (models, pipeline, spatial
    engine) is still cached in `get_query_dispatcher`, and the pending-query flow
    calls this exactly once per question, so nothing is recomputed on rerun.
    """
    dispatcher = get_query_dispatcher()
    history_list = list(history) if history else None
    return dispatcher.dispatch(query, history=history_list, on_step=_on_step)


#: Markers shown on the baseline (no-query) map. Kept small so the first map
#: paint is fast; a real query narrows the map to just the cited companies.
BASELINE_MAP_MARKER_CAP = 60


@st.cache_data(show_spinner=False, ttl=3600)
def baseline_map_payload() -> tuple:
    """Map records + context for the baseline (no-query) view, cached for 1 hour."""
    result = get_map_service().locate("")
    return list(result.records)[:BASELINE_MAP_MARKER_CAP], result.context.to_dict()


@lru_cache(maxsize=1)
def project_data_dir() -> Path:
    return DEFAULT_GEOJSON_PATH.parent


#: The human-validated question set, used to offer quick-pick prompts in the UI.
_QUESTIONS_CSV = Path(__file__).resolve().parents[3] / "data" / "questions_50.csv"


@st.cache_data(show_spinner=False)
def get_example_questions() -> list[tuple[str, str]]:
    """Return ``[(question_id, question), ...]`` from ``data/questions_50.csv``.

    Empty list when the file is absent so the empty state degrades gracefully.
    """
    if not _QUESTIONS_CSV.exists():
        return []
    df = pd.read_csv(_QUESTIONS_CSV)
    cols = {c.lower(): c for c in df.columns}
    qid_col = cols.get("question_id")
    q_col = cols.get("question")
    if q_col is None:
        return []
    out: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        question = str(row[q_col]).strip()
        if not question or question.lower() == "nan":
            continue
        qid = str(row[qid_col]).strip() if qid_col else f"q{len(out) + 1:03d}"
        out.append((qid, question))
    return out
