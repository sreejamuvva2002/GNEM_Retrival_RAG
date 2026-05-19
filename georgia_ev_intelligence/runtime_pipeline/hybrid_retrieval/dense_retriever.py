"""Dense child retriever wrapper for the isolated hybrid retrieval module."""
from __future__ import annotations

from georgia_ev_intelligence.runtime_pipeline.retrieval.dense_pgvector_retriever import (
    DensePgvectorRetriever,
)
from georgia_ev_intelligence.runtime_pipeline.schemas import RetrievedChildChunk


class DenseChildRetriever:
    """Delegate dense child retrieval to the existing pgvector retriever."""

    def __init__(self, retriever: DensePgvectorRetriever | None = None) -> None:
        self._retriever = retriever or DensePgvectorRetriever()

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChildChunk]:
        return self._retriever.search(query, top_k=top_k)
