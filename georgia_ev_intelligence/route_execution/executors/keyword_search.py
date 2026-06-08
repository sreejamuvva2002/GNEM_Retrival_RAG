"""Keyword search executor: BM25 sparse retrieval over child chunks.

Reuses the existing ``BM25Retriever`` (runtime_pipeline) rather than the
README's Postgres full-text approach, because the child chunks store structured
metadata, not a single ``chunk_text`` column.
"""
from __future__ import annotations

from typing import Any

from . import _search_common as common
from ..schemas import ExecutionResult

DEFAULT_TOP_K = 10


def execute_keyword_search(final_route: dict[str, Any]) -> ExecutionResult:
    from georgia_ev_intelligence.runtime_pipeline.retrieval.bm25_retriever import (
        BM25Retriever,
    )

    query = common.query_text(final_route)
    chunks = BM25Retriever().search(query, top_k=DEFAULT_TOP_K)
    return common.build_chunk_result("keyword_search", chunks)
