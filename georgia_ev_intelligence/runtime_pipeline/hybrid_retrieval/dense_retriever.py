"""Dense child retriever wrapper for the isolated hybrid retrieval module.

WHY THIS FILE EXISTS
--------------------
Adapts the lower-level ``DensePgvectorRetriever`` (in ``runtime_pipeline/retrieval/``)
to the ``ChildRetriever`` Protocol expected by the hybrid retrieval orchestrator.
Like its BM25 counterpart (``hybrid_retrieval/bm25_retriever.py``), this is a
thin adapter keeping the orchestrator decoupled from concrete implementations.

DESIGN PATTERN
--------------
Adapter pattern: ``DenseChildRetriever.retrieve(query, top_k)`` delegates to
``DensePgvectorRetriever.search(query, top_k=top_k)``.

CORRECTNESS NOTE
----------------
Each call to ``DensePgvectorRetriever.search()`` opens a new DB connection to
Neon PostgreSQL, embeds the query with the sentence transformer, and executes a
single pgvector cosine distance query.  Under multi-query retrieval, this
means up to 6 connections per question (original + 5 variations).  The Neon
cloud DB handles connection pooling transparently.
"""
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
