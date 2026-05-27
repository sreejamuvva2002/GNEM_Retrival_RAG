"""Data models local to the isolated hybrid retrieval module.

WHY THIS FILE EXISTS
--------------------
Houses the frozen dataclasses that represent intermediate and final outputs
of the hybrid retrieval pipeline.  Keeping models separate from logic avoids
circular imports and makes the data contracts easy to inspect or test.

MODELS DEFINED
--------------
``RerankedChildChunk``
    Wraps a ``RetrievedChildChunk`` (from the shared schemas) with the
    cross-encoder rerank score and ordinal rank.  Provides pass-through
    properties (``chunk_id``, ``parent_record_id``) so callers do not need
    to navigate the nested ``.child`` attribute.
    NOTE: Child-level reranking is NOT currently active in the standard
    pipeline — the active reranking path is parent-level only.  This class
    is retained for future use or alternative experiment configurations.

``HybridRetrievalResult``
    The return type of ``HybridRetrievalOrchestrator.retrieve_with_sources()``.
    Bundles the final reranked ``parent_contexts`` list alongside the raw
    sparse and dense child lists (for diagnostics/tracing) and a
    ``HybridRetrievalTrace`` summary.

``HybridRetrievalTrace``
    Immutable count-level diagnostics for a single retrieval invocation.
    Captures: sparse_child_count, dense_child_count, merged_child_result_count,
    unique_child_chunk_count, unique_parent_id_count,
    parent_context_count_before_rerank, parent_context_count_after_rerank.
    Stored in the JSONL output row (fields prefixed with trace_*) so post-hoc
    analysis can verify multi-query retrieval is working as expected.
"""
from __future__ import annotations

from dataclasses import dataclass

from georgia_ev_intelligence.runtime_pipeline.schemas import (
    ParentContext,
    RetrievedChildChunk,
)


@dataclass(frozen=True)
class RerankedChildChunk:
    """A child chunk scored by the cross-encoder reranker."""

    child: RetrievedChildChunk
    rerank_score: float
    rank: int

    @property
    def chunk_id(self) -> str:
        return self.child.chunk_id

    @property
    def parent_record_id(self) -> str:
        return self.child.parent_record_id


@dataclass(frozen=True)
class HybridRetrievalResult:
    """Retrieval output with source-specific child traces."""

    parent_contexts: list[ParentContext]
    dense_children: list[RetrievedChildChunk]
    sparse_children: list[RetrievedChildChunk]
    trace: "HybridRetrievalTrace | None" = None


@dataclass(frozen=True)
class HybridRetrievalTrace:
    """Deterministic count summary for the active hybrid retrieval flow."""

    sparse_child_count: int
    dense_child_count: int
    merged_child_result_count: int
    unique_child_chunk_count: int
    unique_parent_id_count: int
    parent_context_count_before_rerank: int
    parent_context_count_after_rerank: int
