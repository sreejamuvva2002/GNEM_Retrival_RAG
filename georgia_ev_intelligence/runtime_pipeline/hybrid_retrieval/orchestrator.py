"""Orchestrator for parallel child retrieval, reranking, and parent expansion."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Sequence

from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext

from .config import HybridRetrievalConfig
from .interfaces import ChildReranker, RetrieverStage
from .merger import ChildResultMerger
from .models import RetrieverResultSet
from .parent_mapper import ParentChildMapper


class HybridRetrievalOrchestrator:
    """Clean entry point for the isolated three-stage retrieval flow."""

    def __init__(
        self,
        retrievers: Sequence[RetrieverStage],
        reranker: ChildReranker,
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
        retrieval_results = self._retrieve_children(query)
        merged_children = self._merger.merge(retrieval_results)
        reranked_children = self._reranker.rerank(
            query=query,
            children=merged_children,
            top_k=self._config.reranker_top_k,
        )
        return self._parent_mapper.map_to_parents(
            reranked_children=reranked_children,
            retrieval_results_by_name={
                result.name: result.children for result in retrieval_results
            },
        )

    def _retrieve_children(self, query: str) -> list[RetrieverResultSet]:
        if not self._retrievers:
            return []

        with ThreadPoolExecutor(max_workers=len(self._retrievers)) as executor:
            futures = {
                executor.submit(
                    stage.retriever.retrieve,
                    query,
                    self._config.retriever_top_k,
                ): index
                for index, stage in enumerate(self._retrievers)
            }

            results: list[RetrieverResultSet | None] = [None] * len(self._retrievers)
            for future in as_completed(futures):
                index = futures[future]
                stage = self._retrievers[index]
                results[index] = RetrieverResultSet(
                    name=stage.name,
                    children=future.result(),
                )

        return [result for result in results if result is not None]
