"""Data models local to the isolated hybrid retrieval module."""
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
    # Highest cross-encoder rerank score among the returned parents. None when no
    # parents were returned or the reranker did not expose scores. Used by the
    # self-healing loop as a retrieval-confidence signal.
    top_rerank_score: float | None = None
