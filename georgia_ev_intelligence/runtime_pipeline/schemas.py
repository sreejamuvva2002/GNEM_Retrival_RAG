"""Shared runtime data models for the hybrid retrieval pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChildChunk:
    """A single child chunk returned by dense or BM25 retrieval."""
    chunk_id: str
    parent_record_id: str
    chunk_type: str
    source_row_id: int
    metadata: dict[str, Any]
    score: float


@dataclass
class FusedChildChunk:
    """A child chunk after RRF fusion with combined scoring."""
    chunk_id: str
    parent_record_id: str
    chunk_type: str
    source_row_id: int
    metadata: dict[str, Any]
    rrf_score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


@dataclass
class ParentContext:
    """A parent chunk fetched for answer generation."""
    record_id: str
    source_row_id: int
    parent_chunk_text: str
    metadata: dict[str, Any]
    # Retrieval signals
    max_rrf_score: float
    matched_child_ids: list[str] = field(default_factory=list)
    matched_child_types: list[str] = field(default_factory=list)
    dense_hit_count: int = 0
    bm25_hit_count: int = 0
    combined_score: float = 0.0


@dataclass
class Citation:
    """A citation linking an answer claim back to a source record."""
    citation_id: str  # e.g., "S1"
    parent_record_id: str
    source_row_id: int
    company: str = ""
    retrieval_score: float = 0.0


@dataclass
class CitationOutput:
    """Citations split into used-by-LLM and all-available."""
    used_citations: list[Citation] = field(default_factory=list)
    all_source_records: list[Citation] = field(default_factory=list)


@dataclass
class RetrievalTrace:
    """Full trace of one RAG pipeline run for debugging and evaluation."""
    question: str = ""
    dense_results: list[dict] = field(default_factory=list)
    bm25_results: list[dict] = field(default_factory=list)
    fused_results: list[dict] = field(default_factory=list)
    selected_parent_ids: list[str] = field(default_factory=list)
    fetched_parent_count: int = 0
    llm_context_parent_count: int = 0
    parent_chunk_texts: list[str] = field(default_factory=list)
    context_sent_to_llm: str = ""
    final_answer: str = ""
    citations: list[dict] = field(default_factory=list)
    latency: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    dense_result_count: int = 0
    bm25_result_count: int = 0
    hybrid_result_count: int = 0


@dataclass
class RagResult:
    """Final output of the runtime pipeline."""
    question: str
    answer: str
    citations: CitationOutput = field(default_factory=CitationOutput)
    parent_contexts_used: int = 0
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)


@dataclass
class PipelineConfig:
    """Runtime pipeline configuration with sensible defaults."""
    dense_top_k: int = 100
    bm25_top_k: int = 100
    fused_child_top_k: int = 100
    parent_top_k: int = 60
    rrf_k: int = 60
    multi_child_bonus_2: float = 0.01
    multi_child_bonus_3_plus: float = 0.02
