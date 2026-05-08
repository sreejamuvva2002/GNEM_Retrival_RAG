"""
Phase 4 — Qdrant search helper.

Thin wrapper around `phase2_embedding.vector_store` (search_hybrid,
search_dense, scroll_points) that:

  - translates a deterministic `Entities` object plus an ambiguity-branch
    filter overlay into Qdrant payload-pushdown filters (only safe exact-
    match fields — county/city/industry_group/facility_type/classification_
    method/supplier_affiliation_type/ev_battery_relevant/employment range),
  - emits Candidate objects keyed by `company_row_id` (or canonical name as
    fallback) so the fusion layer can merge them with SQL/Cypher candidates,
  - keeps the existing `_load_all_companies` cached scroll helper available
    for fallback paths.

Why the pushdown is conservative: payload index `MatchValue` is exact, so
pushing down a tier value would over-filter (an exact-tier predicate would
exclude compound-tier rows). Tier and OEM/role matches are deferred to
evidence_validator, which has variant-tolerant `_tier_matches` and OEM
substring logic.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from phase2_embedding.embedder import embed_single
from phase2_embedding.vector_store import (
    scroll_points,
    search_dense as _vs_dense,
    search_hybrid as _vs_hybrid,
)
from phase4_agent.entity_extractor import Entities
from phase4_agent.retrieval_types import Candidate
from shared.logger import get_logger

logger = get_logger("phase4.qdrant_search")


_BASE_COMPANY_FILTERS = {
    "chunk_type": "company",
    "source_type": "gnem_excel",
}
_MASTER_COMPANY_FILTERS = {
    **_BASE_COMPANY_FILTERS,
    "chunk_view": "master",
}

# Maximum candidates fetched per Qdrant search before fusion + reranker.
DEFAULT_K = 120


# ── Payload mapping ──────────────────────────────────────────────────────────

def _safe_pushdown_filters(entities: Entities, branch_filters: dict | None) -> dict[str, Any]:
    """
    Build the dict of payload filters that are safe to push down to Qdrant.

    Excluded on purpose:
      - tier        : exact MatchValue would over-filter compound tiers
      - primary_oems: exact MatchValue, but DB stores comma-separated lists
      - ev_supply_chain_role : multi-role lists need OR, not exact
      - company_name (unless explicitly specified)

    The evidence_validator handles all of the above with variant-tolerant
    logic so candidates are not silently dropped at retrieval time.
    """
    f: dict[str, Any] = dict(_BASE_COMPANY_FILTERS)

    if entities.county:
        f["location_county"] = entities.county
    if entities.industry_group:
        f["industry_group"] = entities.industry_group
    if entities.facility_type:
        f["facility_type"] = entities.facility_type
    if entities.classification_method:
        f["classification_method"] = entities.classification_method
    if entities.supplier_affiliation_type:
        f["supplier_affiliation_type"] = entities.supplier_affiliation_type
    if entities.ev_relevance_value:
        f["ev_battery_relevant"] = entities.ev_relevance_value
    if entities.min_employment is not None:
        f["min_employment"] = float(entities.min_employment)
    if entities.max_employment is not None:
        f["max_employment"] = float(entities.max_employment)

    if branch_filters:
        for k, v in branch_filters.items():
            # Only overlay keys that _build_metadata_filter understands.
            if k in {
                "location_county", "location_city", "industry_group",
                "facility_type", "classification_method",
                "supplier_affiliation_type", "ev_battery_relevant",
                "min_employment", "max_employment",
            } and v is not None:
                f[k] = v

    return f


# ── Result wrapping ──────────────────────────────────────────────────────────

def _payload_to_candidate_row(payload: dict[str, Any]) -> dict[str, Any]:
    """Project a Qdrant payload onto the canonical company-row shape."""
    return {
        "company_row_id":  payload.get("company_row_id"),
        "company_name":    payload.get("company_name", ""),
        "tier":            payload.get("tier", ""),
        "ev_supply_chain_role": payload.get("ev_supply_chain_role", ""),
        "primary_oems":    payload.get("primary_oems", ""),
        "ev_battery_relevant": payload.get("ev_battery_relevant", ""),
        "industry_group":  payload.get("industry_group", ""),
        "facility_type":   payload.get("facility_type", ""),
        "location_city":   payload.get("location_city", ""),
        "location_county": payload.get("location_county", ""),
        "employment":      payload.get("employment"),
        "products_services": payload.get("products_services_full") or payload.get("products_services", ""),
        "classification_method": payload.get("classification_method", ""),
        "supplier_affiliation_type": payload.get("supplier_affiliation_type", ""),
        "chunk_view":      payload.get("chunk_view", ""),
        "source_row_hash": payload.get("source_row_hash", ""),
        "text": (
            payload.get("company_context_text")
            or payload.get("master_text")
            or payload.get("text", "")
        ),
    }


def _wrap_results(results: list[dict[str, Any]], source_name: str) -> list[Candidate]:
    """
    Convert Qdrant search results into Candidate objects.

    Multiple chunks can map to the same canonical company; the fusion
    layer merges by canonical key. Here we keep one Candidate per result
    chunk and let retrieval_fusion.merge collapse duplicates.
    """
    out: list[Candidate] = []
    for r in results:
        payload = r.get("metadata") or {}
        row = _payload_to_candidate_row(payload)
        name = (row.get("company_name") or r.get("company_name") or "").strip()
        if not name:
            continue
        cand = Candidate(
            canonical_name=name,
            company_row_id=row.get("company_row_id") or None,
            source_row_hash=row.get("source_row_hash") or None,
            row=row,
        )
        cand.add_source(source_name, float(r.get("score") or 0.0))
        out.append(cand)
    return out


# ── Public API ───────────────────────────────────────────────────────────────

def dense(query_text: str, entities: Entities, branch_filters: dict | None = None,
          k: int = DEFAULT_K) -> list[Candidate]:
    """Dense-vector search with safe payload pushdown."""
    if not query_text.strip():
        return []
    qfilter = _safe_pushdown_filters(entities, branch_filters)
    qvec = embed_single(query_text)
    if qvec is None:
        logger.warning("dense: embed_single returned None for query=%r", query_text[:80])
        return []
    results = _vs_dense(qvec, top_k=k, filters=qfilter)
    return _wrap_results(results, "dense")


def hybrid(query_text: str, entities: Entities, branch_filters: dict | None = None,
           k: int = DEFAULT_K) -> list[Candidate]:
    """Hybrid dense+sparse RRF search with safe payload pushdown.

    Use this when you want a single fused score blending semantic and BM25
    matching. The two sub-scores are rolled into one via Qdrant's RRF and
    surfaced under the source name 'hybrid'. retrieval_fusion treats this
    as a stand-in for both 'dense' and 'sparse'.
    """
    if not query_text.strip():
        return []
    qfilter = _safe_pushdown_filters(entities, branch_filters)
    qvec = embed_single(query_text)
    if qvec is None:
        logger.warning("hybrid: embed_single returned None for query=%r", query_text[:80])
        return []
    results = _vs_hybrid(query_text, qvec, top_k=k, filters=qfilter)
    return _wrap_results(results, "hybrid")


def sparse(query_text: str, entities: Entities, branch_filters: dict | None = None,
           k: int = DEFAULT_K) -> list[Candidate]:
    """
    Sparse / BM25-style retrieval.

    Note: phase2_embedding.vector_store does not expose a pure-sparse helper
    today; the underlying collection has a sparse index used inside
    search_hybrid via RRF. We implement sparse() as 'hybrid with the same
    query', which gives BM25 plus dense recall and avoids degraded recall
    when a question word never appeared in the dense vocabulary.

    The fusion layer's normalisation handles the score scaling.
    """
    return hybrid(query_text, entities, branch_filters, k=k)


# ── Cached full-scroll fallback ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_all_master_companies() -> list[dict[str, Any]]:
    """Load every master-view company chunk (one per company) once per session."""
    records = scroll_points(filters=_MASTER_COMPANY_FILTERS, limit=600)
    if not records:
        records = scroll_points(filters=_BASE_COMPANY_FILTERS, limit=600)
    rows = [_payload_to_candidate_row(r["payload"]) for r in records]
    rows = [r for r in rows if r.get("company_name")]
    seen_keys: set[Any] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = row.get("company_row_id") or row.get("company_name")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(row)
    logger.info("load_all_master_companies cached %d rows", len(deduped))
    return deduped
