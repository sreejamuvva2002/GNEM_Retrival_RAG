"""Fetch parent chunks from PostgreSQL using fused child chunk results."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import psycopg2

from ...shared import config
from ..schemas import FusedChildChunk, ParentContext, PipelineConfig, RetrievedChildChunk


_FETCH_PARENTS_SQL = """
SELECT
    record_id, source_row_id, company, category, industry_group,
    updated_location, primary_facility_type, ev_supply_chain_role,
    primary_oems, supplier_or_affiliation_type, employment,
    product_service, ev_battery_relevant, parent_chunk_text
FROM parent_chunks
WHERE record_id = ANY(%s);
"""


def fetch_parents(
    fused_children: list[FusedChildChunk],
    dense_results: list[RetrievedChildChunk],
    bm25_results: list[RetrievedChildChunk],
    pipeline_config: PipelineConfig | None = None,
) -> list[ParentContext]:
    """Fetch parent chunks for fused child results, with enriched scoring.

    Steps:
    1. Extract unique parent_record_ids from fused children
    2. Compute max RRF score, matched children, dense/BM25 hit counts per parent
    3. Apply multi-child bonus
    4. Fetch parent chunk text and metadata from PostgreSQL
    5. Return ParentContext objects sorted by combined_score descending
    """
    cfg = pipeline_config or PipelineConfig()

    # Build per-parent aggregation from fused children
    parent_scores: dict[str, float] = defaultdict(float)
    parent_child_ids: dict[str, list[str]] = defaultdict(list)
    parent_child_types: dict[str, set[str]] = defaultdict(set)

    for child in fused_children:
        pid = child.parent_record_id
        if child.rrf_score > parent_scores[pid]:
            parent_scores[pid] = child.rrf_score
        parent_child_ids[pid].append(child.chunk_id)
        parent_child_types[pid].add(child.chunk_type)

    # Count dense and BM25 hits per parent
    dense_ids_by_parent: dict[str, int] = defaultdict(int)
    bm25_ids_by_parent: dict[str, int] = defaultdict(int)

    dense_chunk_ids = {c.chunk_id for c in dense_results}
    bm25_chunk_ids = {c.chunk_id for c in bm25_results}

    for child in fused_children:
        pid = child.parent_record_id
        if child.chunk_id in dense_chunk_ids:
            dense_ids_by_parent[pid] += 1
        if child.chunk_id in bm25_chunk_ids:
            bm25_ids_by_parent[pid] += 1

    # Compute combined score with multi-child bonus
    parent_combined: dict[str, float] = {}
    for pid, max_score in parent_scores.items():
        unique_types = len(parent_child_types[pid])
        if unique_types >= 3:
            bonus = cfg.multi_child_bonus_3_plus
        elif unique_types >= 2:
            bonus = cfg.multi_child_bonus_2
        else:
            bonus = 0.0
        parent_combined[pid] = max_score + bonus

    # Rank parents by combined score, take top_k
    sorted_parents = sorted(parent_combined, key=lambda pid: parent_combined[pid], reverse=True)
    top_parent_ids = sorted_parents[: cfg.parent_top_k]

    if not top_parent_ids:
        return []

    # Fetch parent records from PostgreSQL
    conn = psycopg2.connect(config.NEON_DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(_FETCH_PARENTS_SQL, (top_parent_ids,))
            rows = cur.fetchall()
    finally:
        conn.close()

    # Build lookup: record_id -> row data
    parent_data: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = row[0]
        parent_data[record_id] = {
            "record_id": record_id,
            "source_row_id": int(row[1]) if row[1] is not None else 0,
            "company": row[2] or "",
            "category": row[3] or "",
            "industry_group": row[4] or "",
            "updated_location": row[5] or "",
            "primary_facility_type": row[6] or "",
            "ev_supply_chain_role": row[7] or "",
            "primary_oems": row[8] or "",
            "supplier_or_affiliation_type": row[9] or "",
            "employment": row[10],
            "product_service": row[11] or "",
            "ev_battery_relevant": row[12] or "",
            "parent_chunk_text": row[13] or "",
        }

    # Build ParentContext objects in ranked order
    contexts: list[ParentContext] = []
    for pid in top_parent_ids:
        data = parent_data.get(pid)
        if data is None:
            continue

        metadata = {
            k: v for k, v in data.items()
            if k not in ("record_id", "source_row_id", "parent_chunk_text")
        }

        contexts.append(ParentContext(
            record_id=data["record_id"],
            source_row_id=data["source_row_id"],
            parent_chunk_text=data["parent_chunk_text"],
            metadata=metadata,
            max_rrf_score=parent_scores[pid],
            matched_child_ids=parent_child_ids[pid],
            matched_child_types=sorted(parent_child_types[pid]),
            dense_hit_count=dense_ids_by_parent.get(pid, 0),
            bm25_hit_count=bm25_ids_by_parent.get(pid, 0),
            combined_score=parent_combined[pid],
        ))

    return contexts
