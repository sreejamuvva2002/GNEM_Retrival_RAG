"""Tests for the PreGenerationContextPackage builder."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator.models import (
    ResolvedQueryContext,
)
from georgia_ev_intelligence.runtime_pipeline.pre_retrieval.runtime_context import (
    PreGenerationContextPackage,
)
from georgia_ev_intelligence.runtime_pipeline.pre_retrieval.context_builder import (
    build_pre_generation_context,
)
from georgia_ev_intelligence.runtime_pipeline.schemas import (
    FusedChildChunk,
    ParentContext,
    PipelineConfig,
    RetrievedChildChunk,
)


# --- Fakes ---


@dataclass
class FakeOrchestratorResult:
    parent_contexts: list[ParentContext]
    fused_children: list[FusedChildChunk]
    dense_results: list[RetrievedChildChunk]
    bm25_results: list[RetrievedChildChunk]


class FakeRetrievalOrchestrator:
    """Fake orchestrator that returns predetermined results."""

    def __init__(
        self,
        parents: list[ParentContext] | None = None,
        fused: list[FusedChildChunk] | None = None,
        dense: list[RetrievedChildChunk] | None = None,
        bm25: list[RetrievedChildChunk] | None = None,
        should_fail: bool = False,
    ) -> None:
        self._parents = parents or []
        self._fused = fused or []
        self._dense = dense or []
        self._bm25 = bm25 or []
        self._should_fail = should_fail
        self.search_queries: list[str] = []

    def search(self, question: str):
        self.search_queries.append(question)
        if self._should_fail:
            raise RuntimeError("Retrieval failed (test)")
        return FakeOrchestratorResult(
            parent_contexts=self._parents,
            fused_children=self._fused,
            dense_results=self._dense,
            bm25_results=self._bm25,
        )


def _resolved_context(query: str = "Test query") -> ResolvedQueryContext:
    return ResolvedQueryContext(
        original_query=query,
        original_analysis=None,
        original_phrase_classification=None,
        target_entity="suppliers",
        operation="list",
    )


def _parent(record_id: str = "P1") -> ParentContext:
    return ParentContext(
        record_id=record_id,
        source_row_id=1,
        parent_chunk_text="Test parent text",
        metadata={"company": "TestCo"},
        max_rrf_score=0.5,
    )


def _fused(chunk_id: str = "C1") -> FusedChildChunk:
    return FusedChildChunk(
        chunk_id=chunk_id,
        parent_record_id="P1",
        chunk_type="identity",
        source_row_id=1,
        metadata={},
        rrf_score=0.5,
    )


def _dense(chunk_id: str = "C1") -> RetrievedChildChunk:
    return RetrievedChildChunk(
        chunk_id=chunk_id,
        parent_record_id="P1",
        chunk_type="identity",
        source_row_id=1,
        metadata={},
        score=0.9,
    )


# --- Tests ---


class TestRetrievalUsesOriginalQuery:
    def test_search_receives_original_query(self) -> None:
        orch = FakeRetrievalOrchestrator(parents=[_parent()], fused=[_fused()])
        ctx = _resolved_context("Show Tier 1/2 suppliers in Georgia")

        pkg = build_pre_generation_context(
            query="Show Tier 1/2 suppliers in Georgia",
            resolved_context=ctx,
            retrieval_orchestrator=orch,
        )
        assert orch.search_queries == ["Show Tier 1/2 suppliers in Georgia"]
        assert pkg.original_query == "Show Tier 1/2 suppliers in Georgia"


class TestNoReranker:
    def test_reranked_results_is_none(self) -> None:
        orch = FakeRetrievalOrchestrator(parents=[_parent()])
        pkg = build_pre_generation_context("q", _resolved_context(), orch)
        assert pkg.reranked_results is None
        assert pkg.retrieval_trace.get("reranker_used") is False


class TestParentRecordsFetched:
    def test_parents_populated(self) -> None:
        parents = [_parent("P1"), _parent("P2")]
        orch = FakeRetrievalOrchestrator(parents=parents, fused=[_fused()])
        pkg = build_pre_generation_context("q", _resolved_context(), orch)
        assert len(pkg.parent_records) == 2
        assert pkg.retrieval_trace["parent_context_count"] == 2


class TestContextPackageFields:
    def test_all_fields_populated(self) -> None:
        parents = [_parent()]
        fused = [_fused()]
        dense = [_dense()]
        bm25 = [_dense("C2")]
        orch = FakeRetrievalOrchestrator(
            parents=parents, fused=fused, dense=dense, bm25=bm25
        )
        ctx = _resolved_context()
        pkg = build_pre_generation_context("q", ctx, orch)

        assert pkg.original_query == "q"
        assert pkg.resolved_query_context is ctx
        assert len(pkg.retrieval_results) == 1
        assert pkg.reranked_results is None
        assert len(pkg.parent_records) == 1
        assert pkg.citations == []
        assert pkg.ready_for_generation is True
        assert "dense_result_count" in pkg.retrieval_trace
        assert "bm25_result_count" in pkg.retrieval_trace
        assert "hybrid_result_count" in pkg.retrieval_trace
        assert "parent_context_count" in pkg.retrieval_trace


class TestRetrievalFailure:
    def test_failure_returns_not_ready(self) -> None:
        orch = FakeRetrievalOrchestrator(should_fail=True)
        pkg = build_pre_generation_context("q", _resolved_context(), orch)
        assert pkg.ready_for_generation is False
        assert pkg.retrieval_trace.get("error") == "retrieval_failed"
        assert pkg.parent_records == []


class TestNoFinalAnswerGeneration:
    def test_no_answer_field_in_package(self) -> None:
        """PreGenerationContextPackage has no answer or LLM generation fields."""
        orch = FakeRetrievalOrchestrator(parents=[_parent()])
        pkg = build_pre_generation_context("q", _resolved_context(), orch)
        assert not hasattr(pkg, "answer")
        assert not hasattr(pkg, "final_answer")
        assert not hasattr(pkg, "llm_response")


class TestEmptyRetrieval:
    def test_no_parents_not_ready(self) -> None:
        orch = FakeRetrievalOrchestrator(parents=[], fused=[])
        pkg = build_pre_generation_context("q", _resolved_context(), orch)
        assert pkg.ready_for_generation is False
        assert pkg.parent_records == []
