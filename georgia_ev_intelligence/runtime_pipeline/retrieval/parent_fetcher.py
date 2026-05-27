"""Fetch parent chunks from PostgreSQL using ordered parent record IDs.

WHY THIS FILE EXISTS
--------------------
Implements the "parent retrieval" step of the parent-child chunking strategy.
Child chunks are retrieved by BM25 / dense search (fine-grained, semantically
focused), then their ``parent_record_id`` pointers are used here to fetch the
full parent records that the LLM will actually read.

PARENT-CHILD DESIGN
--------------------
Each company row in the knowledge base is split at index time into:
  - 1 parent_chunk (the full structured text about the company)
  - 5 child_chunks (identity, product_role, oem_relationship,
    location_employment, classification — each a focused slice)

Retrieval operates on child chunks for precision; the LLM receives parent chunks
for completeness.  This file bridges the two: given child hits, retrieve parents.

KEY BEHAVIOUR
-------------
- ``_dedupe()`` preserves the first-seen order of parent_record_ids so the
  natural retrieval-score ordering from the merger/reranker is maintained.
- Uses a single SQL ``WHERE record_id = ANY(%s)`` batch fetch (no N+1).
- Maintains stable ordering: the result list follows the order in which
  ``parent_record_ids`` were passed in, not the DB row order.

CORRECTNESS CONTRACT
--------------------
- Missing parent IDs (IDs that don't exist in the DB) are silently skipped.
- Returns ``ParentContext`` dataclasses carrying ``record_id``,
  ``source_row_id``, and ``parent_chunk_text``.
- ``parent_chunk_text`` is the exact string passed to the LLM as context.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import psycopg2

from ...shared import config
from ..schemas import ParentContext


_FETCH_PARENTS_SQL = """
SELECT
    record_id, source_row_id, parent_chunk_text
FROM parent_chunks
WHERE record_id = ANY(%s);
"""


def fetch_parents(parent_record_ids: Sequence[str]) -> list[ParentContext]:
    """Fetch parent chunks in the provided parent_record_id order."""
    ordered_parent_ids = _dedupe(parent_record_ids)
    if not ordered_parent_ids:
        return []

    # Fetch parent records from PostgreSQL
    conn = psycopg2.connect(config.NEON_DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(_FETCH_PARENTS_SQL, (ordered_parent_ids,))
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
            "parent_chunk_text": row[2] or "",
        }

    contexts: list[ParentContext] = []
    for pid in ordered_parent_ids:
        data = parent_data.get(pid)
        if data is None:
            continue

        contexts.append(ParentContext(
            record_id=data["record_id"],
            source_row_id=data["source_row_id"],
            parent_chunk_text=data["parent_chunk_text"],
        ))

    return contexts


def _dedupe(values: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
