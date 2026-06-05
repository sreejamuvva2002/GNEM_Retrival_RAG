"""Tests for hybrid retrieval orchestration."""
from __future__ import annotations

from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.config import (
    HybridRetrievalConfig,
)
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.merger import (
    ChildResultMerger,
)
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.orchestrator import (
    HybridRetrievalOrchestrator,
)
from georgia_ev_intelligence.runtime_pipeline.retrieval.parent_fetcher import _dedupe
from georgia_ev_intelligence.runtime_pipeline.schemas import (
    ParentContext,
    RetrievedChildChunk,
)


class FakeRetriever:
    def __init__(self, children: list[RetrievedChildChunk]) -> None:
        self._children = children
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChildChunk]:
        self.calls.append((query, top_k))
        return self._children


class FakeParentMapper:
    def __init__(self) -> None:
        self.mapped_children: list[RetrievedChildChunk] = []

    def map_to_parents(
        self,
        children: list[RetrievedChildChunk],
    ) -> list[ParentContext]:
        self.mapped_children = children
        parents: list[ParentContext] = []
        seen: set[str] = set()
        for child in children:
            if child.parent_record_id in seen:
                continue
            seen.add(child.parent_record_id)
            parents.append(ParentContext(
                record_id=child.parent_record_id,
                source_row_id=len(parents) + 1,
                parent_chunk_text=f"parent {child.parent_record_id}",
            ))
        return parents


class FakeParentReranker:
    def __init__(self) -> None:
        self.received_parents: list[ParentContext] = []
        self.received_top_k = 0

    def rerank_parents(
        self,
        query: str,
        parents: list[ParentContext],
        top_k: int,
    ) -> list[ParentContext]:
        self.received_parents = parents
        self.received_top_k = top_k
        return parents[:top_k]


class ReverseParentReranker(FakeParentReranker):
    def rerank_parents(
        self,
        query: str,
        parents: list[ParentContext],
        top_k: int,
    ) -> list[ParentContext]:
        self.received_parents = parents
        self.received_top_k = top_k
        return list(reversed(parents))[:top_k]


class ScoringParentReranker(FakeParentReranker):
    """Reranker exposing score_parents so the orchestrator captures top score."""

    def score_parents(
        self,
        query: str,
        parents: list[ParentContext],
        top_k: int | None = None,
    ) -> list[tuple[ParentContext, float]]:
        self.received_parents = parents
        self.received_top_k = top_k or 0
        scored = [(parent, 1.0 - index * 0.1) for index, parent in enumerate(parents)]
        return scored if top_k is None else scored[:top_k]


def _child(
    chunk_id: str,
    parent_record_id: str,
) -> RetrievedChildChunk:
    return RetrievedChildChunk(
        chunk_id=chunk_id,
        parent_record_id=parent_record_id,
        chunk_type="identity",
        metadata={},
    )


def test_retrieves_250_children_then_reranks_deduped_parents() -> None:
    sparse = FakeRetriever([
        _child("S1", "P1"),
        _child("S2", "P1"),
        _child("SHARED", "P3"),
    ])
    dense = FakeRetriever([
        _child("SHARED", "P3"),
        _child("D1", "P2"),
    ])
    parent_mapper = FakeParentMapper()
    reranker = FakeParentReranker()
    orchestrator = HybridRetrievalOrchestrator(
        retrievers=(sparse, dense),
        reranker=reranker,
        merger=ChildResultMerger(),
        parent_mapper=parent_mapper,
        config=HybridRetrievalConfig(),
    )

    result = orchestrator.retrieve_with_sources("list all suppliers")

    assert sparse.calls == [("list all suppliers", 250)]
    assert dense.calls == [("list all suppliers", 250)]
    assert [child.chunk_id for child in parent_mapper.mapped_children] == [
        "S1",
        "S2",
        "SHARED",
        "D1",
    ]
    assert [parent.record_id for parent in reranker.received_parents] == [
        "P1",
        "P3",
        "P2",
    ]
    assert reranker.received_top_k == 45
    assert [parent.record_id for parent in result.parent_contexts] == [
        "P1",
        "P3",
        "P2",
    ]
    assert result.trace is not None
    assert result.trace.sparse_child_count == 3
    assert result.trace.dense_child_count == 2
    assert result.trace.merged_child_result_count == 5
    assert result.trace.unique_child_chunk_count == 4
    assert result.trace.unique_parent_id_count == 3
    assert result.trace.parent_context_count_before_rerank == 3
    assert result.trace.parent_context_count_after_rerank == 3


def test_final_output_contains_only_reranked_top_k_parent_contexts() -> None:
    sparse = FakeRetriever([
        _child("S1", "P1"),
        _child("S2", "P2"),
    ])
    dense = FakeRetriever([
        _child("D1", "P3"),
    ])
    parent_mapper = FakeParentMapper()
    reranker = ReverseParentReranker()
    orchestrator = HybridRetrievalOrchestrator(
        retrievers=(sparse, dense),
        reranker=reranker,
        merger=ChildResultMerger(),
        parent_mapper=parent_mapper,
        config=HybridRetrievalConfig(reranker_top_k=2),
    )

    result = orchestrator.retrieve_with_sources("rank parent contexts")

    assert [parent.record_id for parent in reranker.received_parents] == [
        "P1",
        "P2",
        "P3",
    ]
    assert [parent.record_id for parent in result.parent_contexts] == ["P3", "P2"]
    assert result.trace is not None
    assert result.trace.parent_context_count_before_rerank == 3
    assert result.trace.parent_context_count_after_rerank == 2


def test_per_call_top_k_overrides_config_budgets() -> None:
    sparse = FakeRetriever([_child("S1", "P1")])
    dense = FakeRetriever([_child("D1", "P2")])
    reranker = FakeParentReranker()
    orchestrator = HybridRetrievalOrchestrator(
        retrievers=(sparse, dense),
        reranker=reranker,
        merger=ChildResultMerger(),
        parent_mapper=FakeParentMapper(),
        config=HybridRetrievalConfig(),  # defaults 250 / 45
    )

    orchestrator.retrieve_with_sources("q", retriever_top_k=80, reranker_top_k=65)

    assert sparse.calls == [("q", 80)]
    assert dense.calls == [("q", 80)]
    assert reranker.received_top_k == 65


def test_top_rerank_score_captured_when_reranker_exposes_scores() -> None:
    sparse = FakeRetriever([_child("S1", "P1")])
    dense = FakeRetriever([_child("D1", "P2")])
    orchestrator = HybridRetrievalOrchestrator(
        retrievers=(sparse, dense),
        reranker=ScoringParentReranker(),
        merger=ChildResultMerger(),
        parent_mapper=FakeParentMapper(),
        config=HybridRetrievalConfig(),
    )

    result = orchestrator.retrieve_with_sources("q")

    assert result.trace is not None
    assert result.trace.top_rerank_score == 1.0  # highest score of the first parent


def test_top_rerank_score_is_none_for_scoreless_reranker() -> None:
    orchestrator = HybridRetrievalOrchestrator(
        retrievers=(FakeRetriever([_child("S1", "P1")]),),
        reranker=FakeParentReranker(),  # no score_parents
        merger=ChildResultMerger(),
        parent_mapper=FakeParentMapper(),
        config=HybridRetrievalConfig(),
    )

    result = orchestrator.retrieve_with_sources("q")

    assert result.trace is not None
    assert result.trace.top_rerank_score is None


def test_parent_record_ids_are_deduplicated_in_fetch_order() -> None:
    assert _dedupe(["P1", "P2", "P1", "P3", "P2"]) == ["P1", "P2", "P3"]


def test_child_merger_preserves_retriever_order_without_rrf() -> None:
    merger = ChildResultMerger()

    merged = merger.merge([
        [_child("S1", "P1"), _child("SHARED", "P2")],
        [_child("D1", "P3"), _child("SHARED", "P2")],
    ])

    assert [child.chunk_id for child in merged] == ["S1", "SHARED", "D1"]
