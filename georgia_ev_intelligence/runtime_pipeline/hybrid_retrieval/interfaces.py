"""Narrow interfaces used by the isolated hybrid retrieval orchestrator."""
from __future__ import annotations

from typing import Protocol

from georgia_ev_intelligence.runtime_pipeline.schemas import (
    ParentContext,
    RetrievedChildChunk,
)

class ChildRetriever(Protocol):
    """Retrieve child chunks for a query."""

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChildChunk]:
        """Return at most top_k child chunks."""


class ParentReranker(Protocol):
    """Rerank deduplicated parent chunks for a query."""

    def rerank_parents(
        self,
        query: str,
        parents: list[ParentContext],
        top_k: int,
    ) -> list[ParentContext]:
        """Return the top reranked parent chunks."""
