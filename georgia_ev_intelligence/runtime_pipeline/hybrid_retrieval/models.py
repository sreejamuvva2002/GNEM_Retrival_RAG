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
