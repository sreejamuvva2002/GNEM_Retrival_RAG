"""
Retrieval layer controller / facade.

Provides a clean public API for all retrieval operations: semantic search,
BM25 search, RRF fusion, entity search, filter mask building, and evidence
formatting. Consumers import from here instead of reaching into individual
retrieval submodules.
"""
from __future__ import annotations

from .filters import (
    build_and_mask,
    build_col_mask,
    best_single_filter,
)
from .rag import (
    run as retrieve_with_match,
    rrf_fuse as fuse_results,
    bm25_search,
    build_bm25_index,
    column_targeted_search,
    exact_entity_search,
    RAGResult,
)
from .semantic import (
    SemanticRetriever,
    build_semantic_retriever,
    retriever_backend_label,
)
from .evidence import (
    select as select_evidence,
)

__all__ = [
    # Filters
    "build_and_mask",
    "build_col_mask",
    "best_single_filter",
    # RAG retrieval
    "retrieve_with_match",
    "fuse_results",
    "bm25_search",
    "build_bm25_index",
    "column_targeted_search",
    "exact_entity_search",
    "RAGResult",
    # Semantic retrieval
    "SemanticRetriever",
    "build_semantic_retriever",
    "retriever_backend_label",
    # Evidence
    "select_evidence",
]
