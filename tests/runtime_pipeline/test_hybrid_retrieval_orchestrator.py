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
    ])
    dense = FakeRetriever([
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
        "D1",
    ]
    assert [parent.record_id for parent in reranker.received_parents] == ["P1", "P2"]
    assert reranker.received_top_k == 45
    assert [parent.record_id for parent in result.parent_contexts] == ["P1", "P2"]
