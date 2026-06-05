"""Orchestrator for parallel child retrieval, reranking, and parent expansion."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Sequence

from georgia_ev_intelligence.runtime_pipeline.debug_trace import record_step
from georgia_ev_intelligence.runtime_pipeline.schemas import (
    ParentContext,
    RetrievedChildChunk,
)

from .config import HybridRetrievalConfig
from .interfaces import ChildRetriever, ParentReranker
from .merger import ChildResultMerger
from .models import HybridRetrievalResult, HybridRetrievalTrace
from .parent_mapper import ParentChildMapper


class HybridRetrievalOrchestrator:
    """Entry point for BM25 + dense child retrieval and parent-level reranking."""

    def __init__(
        self,
        retrievers: Sequence[ChildRetriever],
        reranker: ParentReranker,
        merger: ChildResultMerger,
        parent_mapper: ParentChildMapper,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        self._retrievers = tuple(retrievers)
        self._reranker = reranker
        self._merger = merger
        self._parent_mapper = parent_mapper
        self._config = config or HybridRetrievalConfig()

    @property
    def reranker(self) -> ParentReranker:
        """The parent-level reranker (used by the self-healing loop to re-rank
        the union of sub-query results against the original query)."""
        return self._reranker

    def retrieve(self, query: str) -> list[ParentContext]:
        """Return deduplicated parent chunks for a query."""
        return self.retrieve_with_sources(query).parent_contexts

    def retrieve_with_sources(
        self,
        query: str,
        *,
        retriever_top_k: int | None = None,
        reranker_top_k: int | None = None,
    ) -> HybridRetrievalResult:
        """Return reranked parent contexts plus sparse/dense child traces.

        ``retriever_top_k`` / ``reranker_top_k`` override the configured budgets
        for a single call (defaulting to config). The self-healing loop widens
        ``reranker_top_k`` on an "insufficient" retrieval verdict.
        """
        retriever_k = (
            retriever_top_k if retriever_top_k is not None else self._config.retriever_top_k
        )
        reranker_k = (
            reranker_top_k if reranker_top_k is not None else self._config.reranker_top_k
        )
        retrieval_results = self._retrieve_children(query, retriever_k)
        sparse_children = retrieval_results[0] if len(retrieval_results) > 0 else []
        dense_children = retrieval_results[1] if len(retrieval_results) > 1 else []
        merged_children = self._merger.merge(retrieval_results)
        unique_parent_ids = _unique_parent_record_ids(merged_children)
        parent_contexts = self._parent_mapper.map_to_parents(merged_children)
        # Active reranking is parent-level because parent_chunk_text is the
        # evidence unit passed to the LLM.
        reranked_parent_contexts, top_rerank_score = self._rerank_with_score(
            query=query,
            parents=parent_contexts,
            top_k=reranker_k,
        )
        record_step(
            "retrieval",
            status="ok",
            summary=(
                f"sparse={len(sparse_children)} dense={len(dense_children)} → "
                f"{len(merged_children)} unique chunks → {len(parent_contexts)} parents "
                f"→ top {len(reranked_parent_contexts)} (top_score="
                f"{top_rerank_score if top_rerank_score is not None else 'n/a'})"
            ),
            details={
                "query": query,
                "retriever_top_k": retriever_k,
                "reranker_top_k": reranker_k,
                "sparse_child_count": len(sparse_children),
                "dense_child_count": len(dense_children),
                "merged_child_result_count": sum(len(r) for r in retrieval_results),
                "unique_child_chunk_count": len(merged_children),
                "unique_parent_id_count": len(unique_parent_ids),
                "parent_context_count_before_rerank": len(parent_contexts),
                "parent_context_count_after_rerank": len(reranked_parent_contexts),
                "top_rerank_score": top_rerank_score,
                "sparse_child_chunk_ids": [c.chunk_id for c in sparse_children],
                "dense_child_chunk_ids": [c.chunk_id for c in dense_children],
                "reranked_parents": [
                    {"record_id": p.record_id, "source_row_id": p.source_row_id}
                    for p in reranked_parent_contexts
                ],
            },
        )
        return HybridRetrievalResult(
            parent_contexts=reranked_parent_contexts,
            sparse_children=sparse_children,
            dense_children=dense_children,
            trace=HybridRetrievalTrace(
                sparse_child_count=len(sparse_children),
                dense_child_count=len(dense_children),
                merged_child_result_count=sum(len(result) for result in retrieval_results),
                unique_child_chunk_count=len(merged_children),
                unique_parent_id_count=len(unique_parent_ids),
                parent_context_count_before_rerank=len(parent_contexts),
                parent_context_count_after_rerank=len(reranked_parent_contexts),
                top_rerank_score=top_rerank_score,
            ),
        )

    def _rerank_with_score(
        self,
        query: str,
        parents: list[ParentContext],
        top_k: int,
    ) -> tuple[list[ParentContext], float | None]:
        """Rerank parents, capturing the top cross-encoder score when available.

        Falls back to the score-less ``rerank_parents`` for rerankers (e.g. test
        fakes) that don't implement ``score_parents``.
        """
        score_fn = getattr(self._reranker, "score_parents", None)
        if callable(score_fn):
            scored = score_fn(query, parents, top_k)
            reranked = [parent for parent, _score in scored]
            top_score = float(scored[0][1]) if scored else None
            return reranked, top_score
        reranked = self._reranker.rerank_parents(query=query, parents=parents, top_k=top_k)
        return reranked, None

    def _retrieve_children(
        self,
        query: str,
        retriever_top_k: int,
    ) -> list[list[RetrievedChildChunk]]:
        if not self._retrievers:
            return []

        with ThreadPoolExecutor(max_workers=len(self._retrievers)) as executor:
            futures = {
                executor.submit(
                    retriever.retrieve,
                    query,
                    retriever_top_k,
                ): index
                for index, retriever in enumerate(self._retrievers)
            }

            results: list[list[RetrievedChildChunk] | None] = [None] * len(self._retrievers)
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()

        return [result for result in results if result is not None]


def _unique_parent_record_ids(children: Sequence[RetrievedChildChunk]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for child in children:
        if child.parent_record_id in seen:
            continue
        seen.add(child.parent_record_id)
        ordered.append(child.parent_record_id)
    return ordered
