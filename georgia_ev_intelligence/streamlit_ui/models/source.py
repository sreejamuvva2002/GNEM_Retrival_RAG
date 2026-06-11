"""SourceViewModel — display-friendly projection of a ParentContext.

ParentContext (from the hybrid_retrieval pipeline) carries record_id,
source_row_id and parent_chunk_text. The UI source card needs richer fields
(title, location, type). This module bridges the two: we enrich each
ParentContext using the original kb workbook keyed by source_row_id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # reject NaN


def _clean(value: Any) -> Optional[str]:
    """Normalize a workbook cell to a trimmed string, or None if blank/NaN."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    # Integer-like floats (e.g. employment 100.0) read better without the .0
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


SOURCE_TYPE_CATEGORY_HINTS: Dict[str, str] = {
    "OEM": "company",
    "Tier 1": "supply_chain",
    "Tier 1/2": "supply_chain",
    "Tier 2": "supply_chain",
    "Tier 2/3": "supply_chain",
    "Tier 3": "supply_chain",
    "Government": "government",
    "Infrastructure": "infrastructure",
}


@dataclass
class SourceViewModel:
    """What the sources panel renders for one retrieved chunk."""

    id: str
    title: str
    snippet: str
    source_type: str
    location_name: Optional[str]
    rank: int
    rank_score: float
    record_id: str
    source_row_id: int
    parent_chunk_text: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # Extra workbook fields surfaced in the React-style sources grid.
    category: Optional[str] = None
    industry_group: Optional[str] = None
    location: Optional[str] = None
    address: Optional[str] = None
    facility_type: Optional[str] = None
    ev_supply_chain_role: Optional[str] = None
    primary_oems: Optional[str] = None
    supplier_type: Optional[str] = None
    employment: Optional[str] = None
    product_service: Optional[str] = None
    ev_battery_relevant: Optional[str] = None

    @classmethod
    def from_parent_context(
        cls,
        parent: ParentContext,
        rank: int,
        total: int,
        xlsx_lookup: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> "SourceViewModel":
        snippet = (parent.parent_chunk_text or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:237].rstrip() + "..."

        # KB rows: record_id starts with "KB_ROW_" (see offline_pipeline/chunking/parent_chunk.py:154).
        # Web docs: record_id starts with "WEB_" (see offline_pipeline/web_chunk_builder.py:25).
        # We cannot use source_row_id == 0 to distinguish because KB row 0 also has source_row_id=0.
        is_kb = (parent.record_id or "").startswith("KB_ROW_")
        row = (xlsx_lookup or {}).get(int(parent.source_row_id)) if is_kb else None

        extra: Dict[str, Any] = {}
        if row is None:
            title = parent.record_id or "Source"
            source_type = "unknown" if is_kb else "web"
            location_name = None
            latitude = None
            longitude = None
        else:
            title = str(row.get("company") or parent.record_id or "Source")
            category = str(row.get("category") or "").strip()
            source_type = SOURCE_TYPE_CATEGORY_HINTS.get(category, "company")
            location_name = (
                str(row.get("city") or row.get("county") or row.get("location") or "").strip() or None
            )
            latitude = row.get("latitude")
            longitude = row.get("longitude")
            extra = {
                "category": _clean(row.get("category")),
                "industry_group": _clean(row.get("industry_group")),
                "location": _clean(row.get("location")),
                "address": _clean(row.get("address")),
                "facility_type": _clean(row.get("facility_type")),
                "ev_supply_chain_role": _clean(row.get("ev_supply_chain_role")),
                "primary_oems": _clean(row.get("primary_oems")),
                "supplier_type": _clean(row.get("supplier_type")),
                "employment": _clean(row.get("employment")),
                "product_service": _clean(row.get("product_service")),
                "ev_battery_relevant": _clean(row.get("ev_battery_relevant")),
            }

        rank_score = 0.0 if total <= 0 else max(0.0, 1.0 - (rank - 1) / total)
        return cls(
            id=parent.record_id,
            title=title,
            snippet=snippet or "(empty chunk)",
            source_type=source_type,
            location_name=location_name,
            rank=rank,
            rank_score=rank_score,
            record_id=parent.record_id,
            source_row_id=int(parent.source_row_id or 0),
            parent_chunk_text=parent.parent_chunk_text or "",
            latitude=latitude,
            longitude=longitude,
            **extra,
        )

    @classmethod
    def from_evidence_row(cls, row: Dict[str, Any], rank: int, total: int) -> "SourceViewModel":
        """Build a source card from a SQL/geo evidence row (one KB record).

        Unlike a document chunk, a structured row already carries the workbook
        fields, so no xlsx lookup is needed — we project the row directly. Only
        the columns the query actually selected are present, so the card honestly
        shows what backed the answer (e.g. ev_supply_chain_role when filtered by
        role). There is no parent-chunk text for structured rows.
        """
        def g(*keys: str) -> Optional[str]:
            for key in keys:
                cleaned = _clean(row.get(key))
                if cleaned is not None:
                    return cleaned
            return None

        company = g("company") or "Record"
        record_id = g("record_id") or ""
        category = g("category") or ""
        location = g("updated_location", "location", "city", "county")
        rank_score = 0.0 if total <= 0 else max(0.0, 1.0 - (rank - 1) / total)
        return cls(
            id=record_id or company,
            title=company,
            snippet="",
            source_type=SOURCE_TYPE_CATEGORY_HINTS.get(category, "company"),
            location_name=location,
            rank=rank,
            rank_score=rank_score,
            record_id=record_id,
            source_row_id=0,
            parent_chunk_text="",
            latitude=_safe_float(row.get("latitude")),
            longitude=_safe_float(row.get("longitude")),
            category=g("category"),
            industry_group=g("industry_group"),
            location=location,
            address=g("address"),
            facility_type=g("primary_facility_type", "facility_type"),
            ev_supply_chain_role=g("ev_supply_chain_role"),
            primary_oems=g("primary_oems"),
            supplier_type=g("supplier_or_affiliation_type", "supplier_type"),
            employment=g("employment"),
            product_service=g("product_service"),
            ev_battery_relevant=g("ev_battery_relevant"),
        )


@dataclass
class Provenance:
    """Everything the Sources panel needs to explain one answer.

    Separates *evidence* (the records or aggregate rows the answer is grounded
    in) from *method* (the executed SQL), so the UI can present record cards and
    a distinct "Query" section. ``kind`` mirrors ``ChatResult.evidence_kind``.
    """

    sources: List["SourceViewModel"] = field(default_factory=list)
    group_rows: List[Dict[str, Any]] = field(default_factory=list)
    sql_queries: List[Dict[str, str]] = field(default_factory=list)
    kind: str = ""

    @property
    def count(self) -> int:
        """Number of records (or aggregate rows) backing the answer."""
        return len(self.sources) or len(self.group_rows)

    def has_content(self) -> bool:
        return bool(self.sources or self.group_rows or self.sql_queries)
