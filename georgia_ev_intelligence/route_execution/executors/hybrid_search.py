"""Hybrid search executor: BM25 + dense, fused with Reciprocal Rank Fusion.

Runs both retrievers, deduplicates by ``chunk_id``, fuses the two ranked lists
with RRF, and returns the top chunks mapped to their parent records. RRF is
implemented locally (the existing merger only deduplicates).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _search_common as common
from ..schemas import ExecutionResult

if TYPE_CHECKING:
    from georgia_ev_intelligence.runtime_pipeline.schemas import RetrievedChildChunk

CANDIDATE_TOP_K = 30
FINAL_TOP_K = 10
RRF_K = 60  # standard reciprocal-rank-fusion damping constant


def reciprocal_rank_fusion(
    ranked_lists: list[list["RetrievedChildChunk"]],
    *,
    k: int = RRF_K,
    top_k: int = FINAL_TOP_K,
) -> list["RetrievedChildChunk"]:
    """Fuse several ranked chunk lists into one, ordered by RRF score."""
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, "RetrievedChildChunk"] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_by_id.setdefault(chunk.chunk_id, chunk)

    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunk_by_id[cid] for cid in ordered_ids[:top_k]]


def execute_hybrid_search(final_route: dict[str, Any]) -> ExecutionResult:
    from georgia_ev_intelligence.runtime_pipeline.retrieval.bm25_retriever import (
        BM25Retriever,
    )
    from georgia_ev_intelligence.runtime_pipeline.retrieval.dense_pgvector_retriever import (
        DensePgvectorRetriever,
    )

    query = common.query_text(final_route)
    bm25 = BM25Retriever().search(query, top_k=CANDIDATE_TOP_K)
    dense = DensePgvectorRetriever().search(query, top_k=CANDIDATE_TOP_K)

    fused = reciprocal_rank_fusion([bm25, dense])
    return common.build_chunk_result("hybrid_search", fused)
