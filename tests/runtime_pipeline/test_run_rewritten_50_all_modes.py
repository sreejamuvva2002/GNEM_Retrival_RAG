"""Tests for the all-modes rewritten-question batch runner."""
from __future__ import annotations

import pandas as pd

from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.models import (
    HybridRetrievalResult,
)
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50 import (
    QuestionRow,
)
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50_all_modes import (
    NO_CONTEXT_OUTPUT_COLUMNS,
    OUTPUT_COLUMNS,
    PipelineOutputSpec,
    PipelineSet,
    PipelineWorkbookWriter,
    Rewritten50AllModesRunner,
)
from georgia_ev_intelligence.runtime_pipeline.schemas import (
    ParentContext,
    RetrievedChildChunk,
)


class FakeContextualPipeline:
    def answer(
        self,
        question: str,
        retrieved_context: str,
        timeout: int = 180,
    ) -> str:
        return f"context answer for {question}: {retrieved_context}"


class FakeNonContextualPipeline:
    def answer(self, question: str, timeout: int = 180) -> str:
        return f"no context answer for {question}"


class FakeRetrievalPipeline:
    def retrieve_with_sources(self, question: str) -> HybridRetrievalResult:
        return HybridRetrievalResult(
            parent_contexts=[
                ParentContext(
                    record_id="P1",
                    source_row_id=1,
                    parent_chunk_text=f"final parent context for {question}",
                )
            ],
            dense_children=[
                RetrievedChildChunk(
                    chunk_id="D1",
                    parent_record_id="P1",
                    chunk_type="identity",
                    metadata={"Company": "Dense Co"},
                )
            ],
            sparse_children=[
                RetrievedChildChunk(
                    chunk_id="S1",
                    parent_record_id="P2",
                    chunk_type="operations",
                    metadata={"Company": "Sparse Co"},
                )
            ],
        )


def test_runner_populates_dense_and_sparse_contexts() -> None:
    runner = Rewritten50AllModesRunner(
        retrieval_pipeline_factory=FakeRetrievalPipeline,
        pipelines=PipelineSet(
            only_rag=FakeContextualPipeline(),
            only_pretrained=FakeNonContextualPipeline(),
            rag_plus_pretrained=FakeContextualPipeline(),
        ),
    )

    rows = runner.run([
        QuestionRow(
            serial_number=1,
            question="Which suppliers are in Georgia?",
            golden_answer="Golden",
        )
    ])

    assert rows[0].retrieved_context == "final parent context for Which suppliers are in Georgia?"
    assert "Dense Co" in rows[0].dense_retrieved_context
    assert "Sparse Co" in rows[0].sparse_retrieved_context


def test_workbooks_include_dense_and_sparse_columns(tmp_path) -> None:
    runner = Rewritten50AllModesRunner(
        retrieval_pipeline_factory=FakeRetrievalPipeline,
        pipelines=PipelineSet(
            only_rag=FakeContextualPipeline(),
            only_pretrained=FakeNonContextualPipeline(),
            rag_plus_pretrained=FakeContextualPipeline(),
        ),
    )
    row = runner.run([
        QuestionRow(serial_number=1, question="q", golden_answer="a")
    ])[0]
    writer = PipelineWorkbookWriter(
        output_dir=tmp_path,
        specs=(
            PipelineOutputSpec(key="only_rag", filename="only_rag.xlsx"),
            PipelineOutputSpec(
                key="only_pretrained",
                filename="only_pre_trained.xlsx",
                include_retrieved_context=False,
            ),
        ),
    )

    writer.append(row)

    only_rag = pd.read_excel(tmp_path / "only_rag.xlsx")
    only_pretrained = pd.read_excel(tmp_path / "only_pre_trained.xlsx")
    assert list(only_rag.columns) == OUTPUT_COLUMNS
    assert list(only_pretrained.columns) == NO_CONTEXT_OUTPUT_COLUMNS
    assert only_rag.loc[0, "dense retrieved context"].find("Dense Co") >= 0
    assert only_pretrained.loc[0, "sparse retrieved context"].find("Sparse Co") >= 0
