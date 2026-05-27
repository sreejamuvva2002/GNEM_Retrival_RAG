"""Orchestrator for parallel child retrieval, parent mapping, and reranking.

WHY THIS FILE EXISTS
--------------------
``HybridRetrievalOrchestrator`` is the central coordinator of the retrieval
pipeline.  It wires together BM25 retrieval, dense retrieval, child deduplication,
parent fetching, and cross-encoder reranking into a single coherent flow.

PIPELINE FLOW
-------------
For a single query (``retrieve`` / ``retrieve_with_sources``):
  1. Run BM25 retriever and dense retriever IN PARALLEL via ThreadPoolExecutor.
  2. Merge all child hit lists with ``ChildResultMerger`` (dedup by chunk_id).
  3. Map deduped children to their parent records via ``ParentChildMapper``
     (batch SQL fetch, order-preserving dedup).
  4. Rerank parent chunks with ``CrossEncoderReranker.rerank_parents()``
     using the original query, keeping top ``config.reranker_top_k`` parents.
  5. Return the reranked parent list (+ trace metadata).

For multi-query (``retrieve_multi_query_with_sources``):
  1. For each query in [original_question, variation_1, ..., variation_5]:
     - Run BM25 + dense in parallel with ``per_query_top_k=150``.
     - Accumulate all child hit lists.
  2. Merge all accumulated child lists (dedup by chunk_id across all queries).
  3. Map → Rerank with the ORIGINAL question as anchor.
  Steps 2-5 are otherwise identical to the single-query flow.

PARALLELISM
-----------
``_retrieve_children_top_k`` uses ``ThreadPoolExecutor`` with ``max_workers``
equal to the number of retrievers (2: BM25 + dense).  Both retrievers run
concurrently; results are collected preserving insertion-index order so BM25
results always come first in the merged output regardless of which finishes
first.

CORRECTNESS CONTRACT
--------------------
- The reranker always uses ``queries[0]`` (the original question) as the
  reranking anchor, never a variation.
- ``parent_context_count_after_rerank`` in the trace should be ≤ ``reranker_top_k``
  and reflects exactly what the LLM will receive.
- Trace counts under multi-query mode: ``sparse_child_count`` and
  ``dense_child_count`` are cumulative totals across all queries; this is
  intentional for diagnosing retrieval breadth.
"""
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

    def retrieve(self, query: str) -> list[ParentContext]:
        """Return deduplicated parent chunks for a query."""
        return self.retrieve_with_sources(query).parent_contexts

    def retrieve_with_sources(self, query: str) -> HybridRetrievalResult:
        """Return reranked parent contexts plus sparse/dense child traces."""
        retrieval_results = self._retrieve_children(query)
        sparse_children = retrieval_results[0] if len(retrieval_results) > 0 else []
        dense_children = retrieval_results[1] if len(retrieval_results) > 1 else []
        merged_children = self._merger.merge(retrieval_results)
        unique_parent_ids = _unique_parent_record_ids(merged_children)
        parent_contexts = self._parent_mapper.map_to_parents(merged_children)
        # Active reranking is parent-level because parent_chunk_text is the
        # evidence unit passed to the LLM.
        reranked_parent_contexts = self._reranker.rerank_parents(
            query=query,
            parents=parent_contexts,
            top_k=self._config.reranker_top_k,
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
            ),
        )

    def retrieve_multi_query_with_sources(
        self,
        queries: list[str],
        per_query_top_k: int = 150,
    ) -> HybridRetrievalResult:
        """Run retrieval for every query, merge all children, then rerank parents.

        Each query generates up to ``per_query_top_k`` BM25 hits and
        ``per_query_top_k`` dense hits.  All child chunks are merged and
        deduplicated before being mapped to parent records.  The cross-encoder
        reranker is applied only to parent chunks, using the **first** query
        (the original question) as the reranking anchor.

        Args:
            queries: Non-empty list of query strings.  ``queries[0]`` is treated
                as the original question for reranking.
            per_query_top_k: Maximum children to fetch per retriever per query.

        Returns:
            A :class:`HybridRetrievalResult` with the reranked parent contexts
            and aggregate trace counts.
        """
        if not queries:
            raise ValueError("queries must be non-empty")

        all_retrieval_results: list[list[RetrievedChildChunk]] = []
        total_sparse = 0
        total_dense = 0

        for query in queries:
            per_query_results = self._retrieve_children_top_k(query, per_query_top_k)
            all_retrieval_results.extend(per_query_results)
            if len(per_query_results) > 0:
                total_sparse += len(per_query_results[0])
            if len(per_query_results) > 1:
                total_dense += len(per_query_results[1])

        merged_children = self._merger.merge(all_retrieval_results)
        unique_parent_ids = _unique_parent_record_ids(merged_children)
        parent_contexts = self._parent_mapper.map_to_parents(merged_children)

        reranked_parent_contexts = self._reranker.rerank_parents(
            query=queries[0],
            parents=parent_contexts,
            top_k=self._config.reranker_top_k,
        )

        return HybridRetrievalResult(
            parent_contexts=reranked_parent_contexts,
            sparse_children=[],
            dense_children=[],
            trace=HybridRetrievalTrace(
                sparse_child_count=total_sparse,
                dense_child_count=total_dense,
                merged_child_result_count=sum(
                    len(result) for result in all_retrieval_results
                ),
                unique_child_chunk_count=len(merged_children),
                unique_parent_id_count=len(unique_parent_ids),
                parent_context_count_before_rerank=len(parent_contexts),
                parent_context_count_after_rerank=len(reranked_parent_contexts),
            ),
        )

    def _retrieve_children(self, query: str) -> list[list[RetrievedChildChunk]]:
        return self._retrieve_children_top_k(query, self._config.retriever_top_k)

    def _retrieve_children_top_k(
        self, query: str, top_k: int
    ) -> list[list[RetrievedChildChunk]]:
        if not self._retrievers:
            return []

        with ThreadPoolExecutor(max_workers=len(self._retrievers)) as executor:
            futures = {
                executor.submit(retriever.retrieve, query, top_k): index
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
