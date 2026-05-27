"""Shared runtime data models used across the entire retrieval pipeline.

WHY THIS FILE EXISTS:
  Defines the two core data transfer objects (DTOs) that flow between
  every retrieval module at runtime.  By placing them in a single file
  that has no heavy dependencies (no psycopg2, no torch), any module
  can import them without triggering the full dependency chain.

MODELS:
  RetrievedChildChunk — a single child-chunk result returned by either
      the BM25 or dense retriever.  Fields:
        chunk_id          : globally-unique ID, e.g. "KB_ROW_0042_abc_IDENTITY"
        parent_record_id  : ID of the parent record this chunk belongs to
        chunk_type        : one of identity / product_role / oem_relationship /
                            location_employment / classification
        metadata          : dict of the KB fields included in this chunk type

  ParentContext — a full parent record retrieved from PostgreSQL for LLM
      consumption.  Fields:
        record_id         : unique parent ID
        source_row_id     : original row number in the KB Excel
        parent_chunk_text : the full formatted text block passed to the LLM

RELATIONSHIPS:
  Produced by:  retrieval/bm25_retriever.py, retrieval/dense_pgvector_retriever.py
  Consumed by:  hybrid_retrieval/merger.py, hybrid_retrieval/orchestrator.py,
                hybrid_retrieval/parent_mapper.py, hybrid_retrieval/reranker.py,
                hybrid_retrieval/run_hybrid_rag.py, run_baseline.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievedChildChunk:
    """A single child chunk returned by dense or BM25 retrieval."""

    chunk_id: str
    parent_record_id: str
    chunk_type: str
    metadata: dict[str, Any]


@dataclass
class ParentContext:
    """A parent chunk fetched for answer generation."""

    record_id: str
    source_row_id: int
    parent_chunk_text: str
