"""Map child chunks back to their deduplicated parent chunks.

WHY THIS FILE EXISTS
--------------------
Bridges the child-retrieval and parent-reranking steps of the pipeline.
After BM25 + dense retrieval returns child chunks, each child's
``parent_record_id`` is used to fetch the complete parent record from
PostgreSQL.  This gives the LLM full company records (not just the small
child slices that were used for retrieval).

PARENT-CHILD CHUNKING RECAP
----------------------------
At index time every knowledge-base row produces:
  - 1 ``parent_chunk`` (full structured text: ~200-400 tokens per company)
  - 5 ``child_chunks`` (focused slices: identity, product_role,
    oem_relationship, location_employment, classification)

We retrieve with child chunks (finer-grained embeddings → better recall), but
the LLM reads parent chunks (richer context → better answer quality).
``ParentChildMapper`` is the step that performs this expansion.

KEY BEHAVIOUR
-------------
- ``map_to_parents()`` calls ``fetch_parents()`` which deduplicates parent IDs
  and issues one batched SQL query (no N+1).
- Ordering is stable: parents appear in the same order as their first child hit
  in the merged list, which reflects retrieval score ordering.

CORRECTNESS CONTRACT
--------------------
- Each company appears AT MOST ONCE in the output even if multiple child chunks
  of that company were retrieved (deduplication in ``fetch_parents``).
- The full ``parent_chunk_text`` is passed to the cross-encoder reranker and
  ultimately to the LLM.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from georgia_ev_intelligence.runtime_pipeline.retrieval.parent_fetcher import fetch_parents
from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext


class ChildWithParentRecordId(Protocol):
    """Child-like retrieval result carrying a parent record id."""

    parent_record_id: str


class ParentChildMapper:
    """Expand child hits to unique parent records using parent_record_id."""

    def map_to_parents(
        self,
        children: Sequence[ChildWithParentRecordId],
    ) -> list[ParentContext]:
        if not children:
            return []

        return fetch_parents([
            child.parent_record_id for child in children
        ])
