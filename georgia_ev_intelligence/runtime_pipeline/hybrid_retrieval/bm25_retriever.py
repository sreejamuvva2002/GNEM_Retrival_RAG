"""BM25 child retriever wrapper for the isolated hybrid retrieval module."""
from __future__ import annotations

from georgia_ev_intelligence.runtime_pipeline.retrieval.bm25_retriever import (
    BM25Retriever,
)
from georgia_ev_intelligence.runtime_pipeline.schemas import RetrievedChildChunk


class BM25ChildRetriever:
    """Delegate BM25 child retrieval to the existing runtime retriever."""

    def __init__(self, retriever: BM25Retriever | None = None) -> None:
        self._retriever = retriever or BM25Retriever()

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChildChunk]:
        return self._retriever.search(query, top_k=top_k)
