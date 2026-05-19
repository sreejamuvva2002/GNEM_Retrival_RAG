"""Map reranked child chunks back to deduplicated parent chunks."""
from __future__ import annotations

from collections.abc import Mapping

from georgia_ev_intelligence.runtime_pipeline.retrieval.parent_fetcher import fetch_parents
from georgia_ev_intelligence.runtime_pipeline.schemas import (
    FusedChildChunk,
    ParentContext,
    PipelineConfig,
    RetrievedChildChunk,
)

from .models import RerankedChildChunk


class ParentChildMapper:
    """Expand reranked children to their parent records using parent_record_id."""

    def map_to_parents(
        self,
        reranked_children: list[RerankedChildChunk],
        retrieval_results_by_name: Mapping[str, list[RetrievedChildChunk]] | None = None,
    ) -> list[ParentContext]:
        if not reranked_children:
            return []

        dense_results = _results_matching("dense", retrieval_results_by_name)
        bm25_results = _results_matching("bm25", retrieval_results_by_name)
        fused_children = [_to_fused_child(child) for child in reranked_children]

        fetch_config = PipelineConfig(
            parent_top_k=len(fused_children),
            multi_child_bonus_2=0.0,
            multi_child_bonus_3_plus=0.0,
        )
        return fetch_parents(
            fused_children=fused_children,
            dense_results=dense_results,
            bm25_results=bm25_results,
            pipeline_config=fetch_config,
        )


def _to_fused_child(reranked_child: RerankedChildChunk) -> FusedChildChunk:
    child = reranked_child.child
    return FusedChildChunk(
        chunk_id=child.chunk_id,
        parent_record_id=child.parent_record_id,
        chunk_type=child.chunk_type,
        source_row_id=child.source_row_id,
        metadata=child.metadata,
        rrf_score=1.0 / reranked_child.rank,
    )


def _results_matching(
    name_fragment: str,
    retrieval_results_by_name: Mapping[str, list[RetrievedChildChunk]] | None,
) -> list[RetrievedChildChunk]:
    if not retrieval_results_by_name:
        return []

    results: list[RetrievedChildChunk] = []
    for name, children in retrieval_results_by_name.items():
        if name_fragment in name.lower():
            results.extend(children)
    return results
