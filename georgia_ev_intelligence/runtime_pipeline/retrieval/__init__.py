"""Child retrieval and parent fetching primitives.

This package provides the three low-level retrieval components:
  - ``bm25_retriever``           — BM25 sparse search over child_chunks (PostgreSQL)
  - ``dense_pgvector_retriever`` — Vector cosine search via pgvector
  - ``parent_fetcher``           — Batch fetch parent_chunks by record_id

These are consumed by the ``hybrid_retrieval`` package above, which wraps them
into adapters and orchestrates parallel multi-query retrieval with reranking.
"""
