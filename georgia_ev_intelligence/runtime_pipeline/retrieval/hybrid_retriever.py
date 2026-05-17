"""Hybrid retrieval combining dense pgvector + BM25 via Reciprocal Rank Fusion."""
from __future__ import annotations

from collections import defaultdict

from ..schemas import FusedChildChunk, PipelineConfig, RetrievedChildChunk
from .dense_pgvector_retriever import DensePgvectorRetriever
from .bm25_retriever import BM25Retriever


class HybridRetriever:
    """Orchestrates dense + BM25 retrieval and fuses results with RRF."""

    def __init__(self, pipeline_config: PipelineConfig | None = None) -> None:
        self._config = pipeline_config or PipelineConfig()
        self._dense = DensePgvectorRetriever()
        self._bm25 = BM25Retriever()

    def search(
        self,
        query: str,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        fused_top_k: int | None = None,
    ) -> tuple[list[FusedChildChunk], list[RetrievedChildChunk], list[RetrievedChildChunk]]:
        """Run hybrid retrieval.

        Returns:
            (fused_results, dense_results, bm25_results)
        """
        d_top_k = dense_top_k or self._config.dense_top_k
        b_top_k = bm25_top_k or self._config.bm25_top_k
        f_top_k = fused_top_k or self._config.fused_child_top_k

        dense_results = self._dense.search(query, top_k=d_top_k)
        bm25_results = self._bm25.search(query, top_k=b_top_k)

        fused = _reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            k=self._config.rrf_k,
            top_k=f_top_k,
        )

        return fused, dense_results, bm25_results


def _reciprocal_rank_fusion(
    dense_results: list[RetrievedChildChunk],
    bm25_results: list[RetrievedChildChunk],
    k: int = 60,
    top_k: int = 100,
) -> list[FusedChildChunk]:
    """Merge dense and BM25 results using Reciprocal Rank Fusion.

    RRF score for a chunk at rank r: 1 / (k + r + 1), where r is 0-indexed.
    """
    # Accumulate RRF scores per chunk_id
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_data: dict[str, dict] = {}
    dense_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}

    for rank, chunk in enumerate(dense_results):
        rrf_scores[chunk.chunk_id] += 1.0 / (k + rank + 1)
        dense_ranks[chunk.chunk_id] = rank
        if chunk.chunk_id not in chunk_data:
            chunk_data[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "parent_record_id": chunk.parent_record_id,
                "chunk_type": chunk.chunk_type,
                "source_row_id": chunk.source_row_id,
                "metadata": chunk.metadata,
            }

    for rank, chunk in enumerate(bm25_results):
        rrf_scores[chunk.chunk_id] += 1.0 / (k + rank + 1)
        bm25_ranks[chunk.chunk_id] = rank
        if chunk.chunk_id not in chunk_data:
            chunk_data[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "parent_record_id": chunk.parent_record_id,
                "chunk_type": chunk.chunk_type,
                "source_row_id": chunk.source_row_id,
                "metadata": chunk.metadata,
            }

    # Sort by RRF score descending, take top_k
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    top_ids = sorted_ids[:top_k]

    fused: list[FusedChildChunk] = []
    for chunk_id in top_ids:
        data = chunk_data[chunk_id]
        fused.append(FusedChildChunk(
            chunk_id=data["chunk_id"],
            parent_record_id=data["parent_record_id"],
            chunk_type=data["chunk_type"],
            source_row_id=data["source_row_id"],
            metadata=data["metadata"],
            rrf_score=rrf_scores[chunk_id],
            dense_rank=dense_ranks.get(chunk_id),
            bm25_rank=bm25_ranks.get(chunk_id),
        ))

    return fused
