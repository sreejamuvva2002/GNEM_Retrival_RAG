"""Narrow interfaces used by the isolated hybrid retrieval orchestrator."""
from __future__ import annotations

from typing import Protocol

from georgia_ev_intelligence.runtime_pipeline.schemas import RetrievedChildChunk

from .models import RerankedChildChunk


class ChildRetriever(Protocol):
    """Retrieve child chunks for a query."""

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChildChunk]:
        """Return at most top_k child chunks."""


class ChildReranker(Protocol):
    """Rerank retrieved child chunks for a query."""

    def rerank(
        self,
        query: str,
        children: list[RetrievedChildChunk],
        top_k: int,
    ) -> list[RerankedChildChunk]:
        """Return the top reranked child chunks."""
