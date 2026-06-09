"""Tests for the single-question XLSX trace runner."""
from __future__ import annotations

import json

import pandas as pd

from georgia_ev_intelligence.route_generation.route_service import RouteService
from georgia_ev_intelligence.route_generation.run_trace import (
    run_question_trace,
    write_route_json,
    write_trace_workbook,
)
from georgia_ev_intelligence.route_generation.schemas import RawRoute, RouteName
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.models import (
    HybridRetrievalResult,
    HybridRetrievalTrace,
)
from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext, RetrievedChildChunk


class FakeRetrievalPipeline:
    def retrieve_with_sources(self, question: str) -> HybridRetrievalResult:
        child = RetrievedChildChunk("C1", "P1", "identity", {"company": "Example Co"})
        return HybridRetrievalResult(
            parent_contexts=[ParentContext("P1", 1, f"context for {question}")],
            dense_children=[child],
            sparse_children=[],
            trace=HybridRetrievalTrace(0, 1, 1, 1, 1, 1, 1),
        )


def test_writes_route_steps_and_context_sheets(
    tmp_path,
    fixture_metadata,
    fake_llm_factory,
) -> None:
    service = RouteService(
        provider=fixture_metadata,
        llm_router=fake_llm_factory(response=RawRoute(
            route=RouteName.vector_search,
            confidence=0.9,
            query_focus="battery suppliers",
        )),
    )
    trace_run = run_question_trace(
        "Tell me about battery suppliers",
        route_service=service,
        retrieval_pipeline_factory=FakeRetrievalPipeline,
    )
    output = tmp_path / "outputs" / "trace.xlsx"

    write_trace_workbook(trace_run, output)

    workbook = pd.ExcelFile(output)
    assert {"summary", "pipeline_steps", "final_route", "parent_contexts"} <= set(
        workbook.sheet_names
    )
    summary = pd.read_excel(output, sheet_name="summary")
    contexts = pd.read_excel(output, sheet_name="parent_contexts")
    assert summary.loc[0, "route"] == "vector_search"
    assert summary.loc[0, "retrieval_status"] == "completed"
    assert contexts.loc[0, "context"] == "context for Tell me about battery suppliers"


def test_writes_validated_final_route_json(
    tmp_path,
    fixture_metadata,
    fake_llm_factory,
) -> None:
    service = RouteService(
        provider=fixture_metadata,
        llm_router=fake_llm_factory(response=RawRoute(
            route=RouteName.vector_search,
            confidence=0.9,
            query_focus="battery suppliers",
        )),
    )
    trace_run = run_question_trace(
        "Tell me about battery suppliers",
        route_service=service,
        retrieval_pipeline_factory=FakeRetrievalPipeline,
    )
    output = tmp_path / "outputs" / "trace.json"

    write_route_json(trace_run, output)

    route_json = json.loads(output.read_text(encoding="utf-8"))
    assert route_json["question"] == "Tell me about battery suppliers"
    assert route_json["route"] == "vector_search"
    assert route_json["validation_status"] == "valid"
