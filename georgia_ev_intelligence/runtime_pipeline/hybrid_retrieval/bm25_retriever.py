"""BM25 child retriever wrapper for the isolated hybrid retrieval module.

WHY THIS FILE EXISTS
--------------------
Adapts the lower-level ``BM25Retriever`` (in ``runtime_pipeline/retrieval/``)
to the ``ChildRetriever`` Protocol expected by the hybrid retrieval orchestrator.
This thin adapter layer keeps the orchestrator decoupled from the concrete
retriever implementation — you can swap in a different sparse retriever by
changing this file without touching the orchestrator.

DESIGN PATTERN
--------------
Follows the Adapter pattern: ``BM25ChildRetriever.retrieve(query, top_k)``
delegates to ``BM25Retriever.search(query, top_k=top_k)``.  The naming
difference (``retrieve`` vs ``search``) is the only translation needed.

CORRECTNESS NOTE
----------------
The underlying ``BM25Retriever`` is thread-safe (double-checked locking on
first load) so the same ``BM25ChildRetriever`` instance can safely be used
across parallel query threads in the orchestrator's ``ThreadPoolExecutor``.
"""
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
