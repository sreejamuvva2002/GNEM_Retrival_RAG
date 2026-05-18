"""Log retrieval and generation traces for debugging and evaluation."""
from __future__ import annotations

from ..schemas import (
    CitationOutput,
    FusedChildChunk,
    ParentContext,
    RetrievalTrace,
    RetrievedChildChunk,
)


def build_trace(
    question: str,
    dense_results: list[RetrievedChildChunk],
    bm25_results: list[RetrievedChildChunk],
    fused_results: list[FusedChildChunk],
    fetched_parent_count: int,
    included_parents: list[ParentContext],
    context_sent: str,
    answer: str,
    citations: CitationOutput,
    latency: dict[str, float],
    errors: list[str] | None = None,
) -> RetrievalTrace:
    """Build a RetrievalTrace from pipeline stage outputs."""
    return RetrievalTrace(
        question=question,
        dense_results=[
            {
                "chunk_id": c.chunk_id,
                "parent_record_id": c.parent_record_id,
                "chunk_type": c.chunk_type,
                "score": c.score,
            }
            for c in dense_results[:20]
        ],
        bm25_results=[
            {
                "chunk_id": c.chunk_id,
                "parent_record_id": c.parent_record_id,
                "chunk_type": c.chunk_type,
                "score": c.score,
            }
            for c in bm25_results[:20]
        ],
        fused_results=[
            {
                "chunk_id": c.chunk_id,
                "parent_record_id": c.parent_record_id,
                "chunk_type": c.chunk_type,
                "rrf_score": c.rrf_score,
                "dense_rank": c.dense_rank,
                "bm25_rank": c.bm25_rank,
            }
            for c in fused_results[:20]
        ],
        selected_parent_ids=[p.record_id for p in included_parents],
        fetched_parent_count=fetched_parent_count,
        llm_context_parent_count=len(included_parents),
        parent_chunk_texts=[p.parent_chunk_text for p in included_parents],
        context_sent_to_llm=context_sent,
        final_answer=answer,
        citations=[
            {
                "citation_id": c.citation_id,
                "parent_record_id": c.parent_record_id,
                "company": c.company,
            }
            for c in citations.used_citations
        ],
        latency=latency,
        errors=errors or [],
    )
