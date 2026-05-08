"""
Phase 4 — Qdrant search helper.

This module is a candidate-retrieval wrapper around embeddings_store.vector_store.

It provides:
  - dense()  : true dense-vector retrieval using search_dense
  - hybrid() : dense + sparse/RRF retrieval using search_hybrid
  - sparse() : true sparse/BM25 retrieval only if vector_store.search_sparse exists;
               otherwise explicit hybrid fallback with audit metadata

Important:
Qdrant retrieval is only a candidate-recall layer.
Final candidates must still pass:
  - deduplication
  - hard-filter validation
  - reranking
  - evidence validation
before answer generation.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Callable

from embeddings_store.doc_embedder import embed_single
from embeddings_store.vector_store import (
    scroll_points,
    search_dense as _vs_dense,
    search_hybrid as _vs_hybrid,
)

try:
    from embeddings_store.vector_store import search_sparse as _vs_sparse  # type: ignore
except Exception:
    _vs_sparse = None  # type: ignore

from filters_and_validation.query_entity_extractor import Entities
from core_agent.retrieval_types import Candidate
from shared.logger import get_logger

logger = get_logger("retrievals.qdrant_search")


_BASE_COMPANY_FILTERS = {
    "chunk_type": "company",
    "source_type": "gnem_excel",
}

_MASTER_COMPANY_FILTERS = {
    **_BASE_COMPANY_FILTERS,
    "chunk_view": "master",
}

DEFAULT_K = int(os.getenv("QDRANT_SEARCH_DEFAULT_K", "120"))
FULL_SCROLL_LIMIT = int(os.getenv("QDRANT_MASTER_SCROLL_LIMIT", "5000"))


def _get_entity_value(entities: Entities, *names: str) -> Any:
    for name in names:
        value = getattr(entities, name, None)
        if value not in (None, "", [], {}):
            return value
    return None


def _embed_query(query_text: str) -> list[float] | None:
    try:
        qvec = embed_single(query_text)
    except Exception as exc:
        logger.exception("embed_single failed for query=%r: %s", query_text[:120], exc)
        return None

    if qvec is None:
        logger.warning("embed_single returned None for query=%r", query_text[:120])
        return None

    return qvec


def _safe_pushdown_filters(
    entities: Entities,
    branch_filters: dict | None,
) -> dict[str, Any]:
    """
    Build safe Qdrant payload filters.

    Safe pushdown:
      - source_type
      - chunk_type
      - location_county
      - location_city
      - industry_group
      - facility_type
      - classification_method
      - supplier_affiliation_type
      - ev_battery_relevant
      - min_employment
      - max_employment

    Not pushed down:
      - tier
      - primary_oems
      - ev_supply_chain_role
      - products_services
      - company_name by default

    Reason:
    Qdrant MatchValue filtering is exact. Fields like tier/OEM/role often
    contain compound strings, so exact pushdown can wrongly remove valid rows.
    These must be checked later by evidence_validator.
    """
    f: dict[str, Any] = dict(_BASE_COMPANY_FILTERS)

    county = _get_entity_value(entities, "county", "location_county")
    city = _get_entity_value(entities, "city", "location_city")
    industry_group = _get_entity_value(entities, "industry_group")
    facility_type = _get_entity_value(entities, "facility_type")
    classification_method = _get_entity_value(entities, "classification_method")
    supplier_affiliation_type = _get_entity_value(
        entities,
        "supplier_affiliation_type",
        "supplier_type",
    )
    ev_relevance_value = _get_entity_value(
        entities,
        "ev_relevance_value",
        "ev_battery_relevant",
        "ev_relevant",
    )

    if county:
        f["location_county"] = county
    if city:
        f["location_city"] = city
    if industry_group:
        f["industry_group"] = industry_group
    if facility_type:
        f["facility_type"] = facility_type
    if classification_method:
        f["classification_method"] = classification_method
    if supplier_affiliation_type:
        f["supplier_affiliation_type"] = supplier_affiliation_type
    if ev_relevance_value:
        f["ev_battery_relevant"] = ev_relevance_value

    min_employment = _get_entity_value(entities, "min_employment")
    max_employment = _get_entity_value(entities, "max_employment")

    if min_employment is not None:
        f["min_employment"] = float(min_employment)
    if max_employment is not None:
        f["max_employment"] = float(max_employment)

    if branch_filters:
        for k, v in branch_filters.items():
            if k in {
                "location_county",
                "location_city",
                "industry_group",
                "facility_type",
                "classification_method",
                "supplier_affiliation_type",
                "ev_battery_relevant",
                "min_employment",
                "max_employment",
            } and v is not None:
                f[k] = v

    return f


def _payload_to_candidate_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "company_row_id": payload.get("company_row_id"),
        "company_name": payload.get("company_name", ""),
        "tier": payload.get("tier", ""),
        "ev_supply_chain_role": payload.get("ev_supply_chain_role", ""),
        "primary_oems": payload.get("primary_oems", ""),
        "ev_battery_relevant": payload.get("ev_battery_relevant", ""),
        "industry_group": payload.get("industry_group", ""),
        "facility_type": payload.get("facility_type", ""),
        "location_city": payload.get("location_city", ""),
        "location_county": payload.get("location_county", ""),
        "employment": payload.get("employment"),
        "products_services": (
            payload.get("products_services_full")
            or payload.get("products_services")
            or ""
        ),
        "classification_method": payload.get("classification_method", ""),
        "supplier_affiliation_type": payload.get("supplier_affiliation_type", ""),
        "chunk_view": payload.get("chunk_view", ""),
        "chunk_type": payload.get("chunk_type", ""),
        "source_type": payload.get("source_type", ""),
        "source_row_hash": payload.get("source_row_hash", ""),
        "kb_schema_version": payload.get("kb_schema_version", ""),
        "text": (
            payload.get("company_context_text")
            or payload.get("master_text")
            or payload.get("parent_text")
            or payload.get("text", "")
        ),
    }


def _wrap_results(
    results: list[dict[str, Any]],
    actual_source_name: str,
    requested_source_name: str | None = None,
) -> list[Candidate]:
    """
    Convert Qdrant results into Candidate objects.

    actual_source_name:
      The actual retriever used: dense, hybrid, sparse.

    requested_source_name:
      The retriever requested by caller. Useful when sparse() falls back to hybrid.
    """
    out: list[Candidate] = []

    for r in results:
        payload = r.get("metadata") or r.get("payload") or {}
        if not isinstance(payload, dict):
            continue

        row = _payload_to_candidate_row(payload)
        name = (row.get("company_name") or r.get("company_name") or "").strip()

        if not name:
            continue

        raw_score = float(r.get("score") or 0.0)

        row["_retrieval_actual_source"] = actual_source_name
        row["_retrieval_requested_source"] = requested_source_name or actual_source_name
        row["_raw_retrieval_score"] = raw_score
        row["_qdrant_point_id"] = (
            r.get("id")
            or r.get("point_id")
            or payload.get("point_id")
            or ""
        )

        cand = Candidate(
            canonical_name=name,
            company_row_id=row.get("company_row_id") or None,
            source_row_hash=row.get("source_row_hash") or None,
            row=row,
        )

        source_label = actual_source_name
        if requested_source_name and requested_source_name != actual_source_name:
            source_label = f"{requested_source_name}_via_{actual_source_name}"

        cand.add_source(source_label, raw_score)
        out.append(cand)

    return out


def _run_search(
    *,
    query_text: str,
    entities: Entities,
    branch_filters: dict | None,
    k: int,
    search_fn: Callable[..., list[dict[str, Any]]],
    actual_source_name: str,
    requested_source_name: str | None = None,
    needs_query_vector: bool,
    pass_query_text: bool,
) -> list[Candidate]:
    if not query_text or not query_text.strip():
        return []

    qfilter = _safe_pushdown_filters(entities, branch_filters)

    qvec: list[float] | None = None
    if needs_query_vector:
        qvec = _embed_query(query_text)
        if qvec is None:
            return []

    try:
        if pass_query_text and needs_query_vector:
            results = search_fn(query_text, qvec, top_k=k, filters=qfilter)
        elif pass_query_text and not needs_query_vector:
            results = search_fn(query_text, top_k=k, filters=qfilter)
        else:
            results = search_fn(qvec, top_k=k, filters=qfilter)
    except Exception as exc:
        logger.exception(
            "Qdrant %s search failed for query=%r: %s",
            actual_source_name,
            query_text[:120],
            exc,
        )
        return []

    return _wrap_results(
        results,
        actual_source_name=actual_source_name,
        requested_source_name=requested_source_name,
    )


def dense(
    query_text: str,
    entities: Entities,
    branch_filters: dict | None = None,
    k: int = DEFAULT_K,
) -> list[Candidate]:
    """
    True dense-vector search.

    Uses:
      embeddings_store.vector_store.search_dense(query_vector, top_k, filters)
    """
    return _run_search(
        query_text=query_text,
        entities=entities,
        branch_filters=branch_filters,
        k=k,
        search_fn=_vs_dense,
        actual_source_name="dense",
        requested_source_name="dense",
        needs_query_vector=True,
        pass_query_text=False,
    )


def hybrid(
    query_text: str,
    entities: Entities,
    branch_filters: dict | None = None,
    k: int = DEFAULT_K,
) -> list[Candidate]:
    """
    True hybrid search.

    Uses:
      embeddings_store.vector_store.search_hybrid(query_text, query_vector, top_k, filters)

    This should represent Qdrant RRF fusion between dense and sparse/BM25.
    """
    return _run_search(
        query_text=query_text,
        entities=entities,
        branch_filters=branch_filters,
        k=k,
        search_fn=_vs_hybrid,
        actual_source_name="hybrid",
        requested_source_name="hybrid",
        needs_query_vector=True,
        pass_query_text=True,
    )


def sparse(
    query_text: str,
    entities: Entities,
    branch_filters: dict | None = None,
    k: int = DEFAULT_K,
) -> list[Candidate]:
    """
    Sparse/BM25 retrieval.

    If vector_store.search_sparse exists:
      uses true sparse search.

    If vector_store.search_sparse does not exist:
      uses hybrid search as an explicit fallback and labels the candidate source
      as sparse_via_hybrid.

    This avoids pretending hybrid fallback is pure sparse retrieval.
    """
    if _vs_sparse is not None:
        return _run_search(
            query_text=query_text,
            entities=entities,
            branch_filters=branch_filters,
            k=k,
            search_fn=_vs_sparse,
            actual_source_name="sparse",
            requested_source_name="sparse",
            needs_query_vector=False,
            pass_query_text=True,
        )

    logger.warning(
        "sparse() requested but embeddings_store.vector_store.search_sparse "
        "is not available. Falling back to hybrid search and labeling source "
        "as sparse_via_hybrid."
    )

    return _run_search(
        query_text=query_text,
        entities=entities,
        branch_filters=branch_filters,
        k=k,
        search_fn=_vs_hybrid,
        actual_source_name="hybrid",
        requested_source_name="sparse",
        needs_query_vector=True,
        pass_query_text=True,
    )


def _scroll_company_records_best_effort(
    filters: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    records = scroll_points(filters=filters, limit=limit)

    if len(records) >= limit:
        logger.warning(
            "scroll_points returned %d records, equal to or above limit=%d. "
            "Fallback company cache may be truncated. Increase "
            "QDRANT_MASTER_SCROLL_LIMIT or implement paginated scroll.",
            len(records),
            limit,
        )

    return records


@lru_cache(maxsize=4)
def load_all_master_companies(limit: int = FULL_SCROLL_LIMIT) -> list[dict[str, Any]]:
    """
    Load all master-view company chunks once per session.

    This is a fallback helper for validation/complete-list paths.
    It is not the preferred semantic retrieval path.
    """
    records = _scroll_company_records_best_effort(
        filters=_MASTER_COMPANY_FILTERS,
        limit=limit,
    )

    if not records:
        records = _scroll_company_records_best_effort(
            filters=_BASE_COMPANY_FILTERS,
            limit=limit,
        )

    rows: list[dict[str, Any]] = []

    for record in records:
        payload = record.get("payload") or record.get("metadata") or {}
        if not isinstance(payload, dict):
            continue

        row = _payload_to_candidate_row(payload)
        if row.get("company_name"):
            rows.append(row)

    seen_keys: set[Any] = set()
    deduped: list[dict[str, Any]] = []

    for row in rows:
        key = (
            row.get("company_row_id")
            or row.get("source_row_hash")
            or row.get("company_name")
        )

        if not key or key in seen_keys:
            continue

        seen_keys.add(key)
        deduped.append(row)

    logger.info("load_all_master_companies cached %d rows", len(deduped))
    return deduped


__all__ = [
    "DEFAULT_K",
    "FULL_SCROLL_LIMIT",
    "dense",
    "hybrid",
    "sparse",
    "load_all_master_companies",
]