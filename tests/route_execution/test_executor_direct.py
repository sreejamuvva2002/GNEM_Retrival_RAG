"""Dispatcher tests for non-retrieval routes (no database access)."""
from __future__ import annotations

from georgia_ev_intelligence.route_execution import execute_route
from georgia_ev_intelligence.route_execution.executors.hybrid_search import (
    reciprocal_rank_fusion,
)
from georgia_ev_intelligence.route_execution.schemas import STATUS_FAILED, STATUS_SUCCESS


def test_no_retrieval():
    result = execute_route({"route": "no_retrieval", "reason": "Simple greeting."})
    assert result.status == STATUS_SUCCESS
    assert result.evidence["type"] == "direct"
    assert "greeting" in result.answer.lower()


def test_clarification_needed_echoes_message():
    result = execute_route({
        "route": "clarification_needed",
        "clarification": {"message": "Which county did you mean?"},
    })
    assert result.status == STATUS_SUCCESS
    assert result.evidence["type"] == "clarification"
    assert result.answer == "Which county did you mean?"


def test_out_of_domain():
    result = execute_route({"route": "out_of_domain"})
    assert result.status == STATUS_SUCCESS
    assert result.evidence["type"] == "out_of_domain"


def test_unsupported_route_fails_gracefully():
    result = execute_route({"route": "teleport"})
    assert result.status == STATUS_FAILED
    assert "Unsupported route" in (result.error or "")


class _Chunk:
    def __init__(self, chunk_id: str):
        self.chunk_id = chunk_id
        self.parent_record_id = f"p_{chunk_id}"
        self.chunk_type = "identity"
        self.metadata = {}


def test_rrf_prefers_items_ranked_high_in_both_lists():
    bm25 = [_Chunk("a"), _Chunk("b"), _Chunk("c")]
    dense = [_Chunk("b"), _Chunk("a"), _Chunk("d")]
    fused = reciprocal_rank_fusion([bm25, dense], top_k=4)
    ids = [c.chunk_id for c in fused]
    # 'a' and 'b' appear high in both -> ranked above 'c'/'d'; no duplicates.
    assert set(ids[:2]) == {"a", "b"}
    assert len(ids) == len(set(ids)) == 4
