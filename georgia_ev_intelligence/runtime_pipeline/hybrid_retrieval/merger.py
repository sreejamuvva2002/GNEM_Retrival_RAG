"""Result merging and deduplication for child chunk retrieval results.

WHY THIS FILE EXISTS
--------------------
After BM25 and dense retrieval each return up to N child chunks, this module
combines them into a single deduplicated list.  Deduplication is necessary
because the same child chunk may appear in both the sparse and dense result
sets (they search the same underlying data).

DEDUPLICATION STRATEGY
-----------------------
- Chunks are deduplicated by ``chunk_id`` (the primary key in ``child_chunks``).
- First-seen ordering is preserved: a chunk that appears in BM25 results before
  dense results retains its BM25-order position.
- Scores are NOT combined or averaged — the ranking is by appearance order.
  The cross-encoder reranker (applied downstream at the parent level) produces
  the final relevance ordering; score merging here would be premature.

MULTI-QUERY USAGE
-----------------
Under multi-query retrieval (original question + up to 5 variations), the
orchestrator calls ``_retrieve_children_top_k`` for each query and collects
multiple lists.  ``ChildResultMerger.merge()`` accepts a variable number of
``Iterable[RetrievedChildChunk]`` iterables so it handles both single-query
(2 iterables: BM25, dense) and multi-query (2 × N_queries iterables) uniformly.

CORRECTNESS CONTRACT
--------------------
- No child chunk is duplicated in the output.
- Output order is deterministic (input order preserved for first occurrences).
- The merger has no state and can be called concurrently.
"""
from __future__ import annotations

from collections.abc import Iterable

from georgia_ev_intelligence.runtime_pipeline.schemas import RetrievedChildChunk


class ChildResultMerger:
    """Merge child results from retrievers and deduplicate by chunk_id."""

    def merge(
        self,
        result_sets: Iterable[Iterable[RetrievedChildChunk]],
    ) -> list[RetrievedChildChunk]:
        merged: list[RetrievedChildChunk] = []
        seen_chunk_ids: set[str] = set()

        for result_set in result_sets:
            for child in result_set:
                if child.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(child.chunk_id)
                merged.append(child)

        return merged
