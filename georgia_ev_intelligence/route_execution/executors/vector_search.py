"""Vector search executor: dense semantic retrieval over child chunks.

Reuses the existing ``DensePgvectorRetriever`` (pgvector cosine search with the
same embedding model used at ingestion time).
"""
from __future__ import annotations

from typing import Any

from . import _search_common as common
from ..schemas import ExecutionResult

DEFAULT_TOP_K = 10


def execute_vector_search(final_route: dict[str, Any]) -> ExecutionResult:
    from georgia_ev_intelligence.runtime_pipeline.retrieval.dense_pgvector_retriever import (
        DensePgvectorRetriever,
    )

    query = common.query_text(final_route)
    chunks = DensePgvectorRetriever().search(query, top_k=DEFAULT_TOP_K)
    return common.build_chunk_result("vector_search", chunks)
