"""Orchestrator for parallel child retrieval, reranking, and parent expansion."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Sequence

from georgia_ev_intelligence.runtime_pipeline.schemas import (
    ParentContext,
    RetrievedChildChunk,
)

from .config import HybridRetrievalConfig
from .interfaces import ChildRetriever, ParentReranker
from .merger import ChildResultMerger
from .models import HybridRetrievalResult
from .parent_mapper import ParentChildMapper


class HybridRetrievalOrchestrator:
    """Clean entry point for the active three-stage retrieval flow."""

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

    def retrieve(self, query: str) -> list[ParentContext]:
        """Return deduplicated parent chunks for a query."""
        return self.retrieve_with_sources(query).parent_contexts

    def retrieve_with_sources(self, query: str) -> HybridRetrievalResult:
        """Return final parents plus dense and sparse child retrieval traces."""
        retrieval_results = self._retrieve_children(query)
        merged_children = self._merger.merge(retrieval_results)
        parent_contexts = self._parent_mapper.map_to_parents(merged_children)
        reranked_parent_contexts = self._reranker.rerank_parents(
            query=query,
            parents=parent_contexts,
            top_k=self._config.reranker_top_k,
        )
        return HybridRetrievalResult(
            parent_contexts=reranked_parent_contexts,
            sparse_children=retrieval_results[0] if len(retrieval_results) > 0 else [],
            dense_children=retrieval_results[1] if len(retrieval_results) > 1 else [],
        )

    def _retrieve_children(self, query: str) -> list[list[RetrievedChildChunk]]:
        if not self._retrievers:
            return []

        with ThreadPoolExecutor(max_workers=len(self._retrievers)) as executor:
            futures = {
                executor.submit(
                    retriever.retrieve,
                    query,
                    self._config.retriever_top_k,
                ): index
                for index, retriever in enumerate(self._retrievers)
            }

            results: list[list[RetrievedChildChunk] | None] = [None] * len(self._retrievers)
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()

        return [result for result in results if result is not None]
