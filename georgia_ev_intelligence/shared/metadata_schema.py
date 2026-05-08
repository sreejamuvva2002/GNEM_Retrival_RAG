"""
Approved hardcoded retrieval grammar for Phase 4 — schema metadata only.

This file is the single allowed home for hardcoded constants used by the
retrieval pipeline. Anything in this file must be either:
  - the *name* of a KB column / Qdrant payload key / Neo4j label / relationship
  - the *name* of a generic operator or query class
  - a numeric weight that controls fusion math

Anything domain-factual (a real company name, OEM brand, county, tier value,
EV role, facility type, product term) is FORBIDDEN here. Those values are
loaded at runtime from Postgres / Neo4j / Qdrant via metadata_loader.py.

If you find yourself adding a literal like "Hyundai", "Tier 1", "Battery
Cell", "Troup County", or "small scale → employment<200" to this file —
stop. That belongs in the database (gev_companies, gev_domain_mapping_rules)
or in the question's runtime entity extraction, not here.
"""
from __future__ import annotations

from typing import Final


# ── Canonical field names (logical key → KB column name) ─────────────────────
CANONICAL_FIELDS: Final[dict[str, str]] = {
    "company":        "company_name",
    "tier":           "tier",
    "role":           "ev_supply_chain_role",
    "product":        "products_services",
    "oem":            "primary_oems",
    "employment":     "employment",
    "county":         "location_county",
    "city":           "location_city",
    "facility_type":  "facility_type",
    "industry_group": "industry_group",
    "ev_relevant":    "ev_battery_relevant",
    "supplier_type":  "supplier_affiliation_type",
}


# ── Field type tags for filter dispatch ──────────────────────────────────────
FIELD_TYPES: Final[dict[str, str]] = {
    "company_name":              "keyword",
    "tier":                      "keyword",
    "ev_supply_chain_role":      "text",
    "products_services":         "text",
    "primary_oems":              "text",
    "employment":                "float",
    "location_county":           "keyword",
    "location_city":             "keyword",
    "facility_type":             "keyword",
    "industry_group":            "keyword",
    "ev_battery_relevant":       "keyword",
    "supplier_affiliation_type": "keyword",
}


# ── Fields that may appear as a hard filter on a candidate ───────────────────
HARD_FILTER_FIELDS: Final[frozenset[str]] = frozenset({
    "tier",
    "company_name",
    "location_county",
    "location_city",
    "primary_oems",
    "ev_supply_chain_role",
    "employment",
    "facility_type",
    "industry_group",
    "supplier_affiliation_type",
    "ev_battery_relevant",
})


# ── Operators allowed in any structured filter ───────────────────────────────
SUPPORTED_OPERATORS: Final[frozenset[str]] = frozenset({
    "equals",
    "contains",
    "in",
    "not_in",
    "greater_than",
    "less_than",
    "between",
})


# ── Coarse query classes (round-trip strings; mirror QueryClass enum) ────────
QUERY_CLASSES: Final[frozenset[str]] = frozenset({
    "exact_filter_query",
    "aggregate_query",
    "count_query",
    "rank_query",
    "top_n_query",
    "network_query",
    "product_capability_query",
    "risk_query",
    "ambiguous_semantic_query",
    "fallback_semantic_query",
})


# ── Final score fusion weights (must sum to 1.0) ─────────────────────────────
FINAL_SCORE_WEIGHTS: Final[dict[str, float]] = {
    "structured_filter_score":    0.40,
    "reranker_score":             0.20,
    "bm25_sparse_score":          0.15,
    "dense_vector_score":         0.15,
    "synonym_mapping_confidence": 0.05,
    "metadata_match_score":       0.05,
}


# ── Multi-view chunking schema for company records in Qdrant ─────────────────
COMPANY_CHUNK_VIEWS: Final[tuple[str, ...]] = (
    "master",
    "role",
    "product",
    "oem",
    "classification",
    "capability",
    "location",
)


# ── Required keys on every Qdrant company-chunk payload ──────────────────────
REQUIRED_QDRANT_PAYLOAD: Final[frozenset[str]] = frozenset({
    "company_name",
    "tier",
    "ev_supply_chain_role",
    "primary_oems",
    "industry_group",
    "location_city",
    "location_county",
    "employment",
    "products_services",
    "source_type",
    "chunk_type",
    "chunk_view",
    "company_row_id",
    "source_row_hash",
    "kb_schema_version",
})


# ── Neo4j graph schema labels and relationships ──────────────────────────────
GRAPH_LABELS: Final[frozenset[str]] = frozenset({
    "Company",
    "Location",
    "OEM",
    "Tier",
    "IndustryGroup",
    "Product",
})

GRAPH_RELATIONSHIPS: Final[frozenset[str]] = frozenset({
    "LOCATED_IN",
    "SUPPLIES_TO",
    "IN_TIER",
    "IN_INDUSTRY",
    "MANUFACTURES",
})


__all__ = [
    "CANONICAL_FIELDS",
    "FIELD_TYPES",
    "HARD_FILTER_FIELDS",
    "SUPPORTED_OPERATORS",
    "QUERY_CLASSES",
    "FINAL_SCORE_WEIGHTS",
    "COMPANY_CHUNK_VIEWS",
    "REQUIRED_QDRANT_PAYLOAD",
    "GRAPH_LABELS",
    "GRAPH_RELATIONSHIPS",
]
