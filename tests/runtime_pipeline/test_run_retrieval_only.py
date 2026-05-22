"""Tests for the retrieval-only rewritten-question batch runner."""
from __future__ import annotations

import pandas as pd

from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.models import (
    HybridRetrievalResult,
)
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50 import (
    QuestionRow,
    _load_questions,
)
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50_retrieval_only import (
    OUTPUT_COLUMNS,
    RetrievalOnlyRunner,
    RetrievalWorkbookWriter,
)
from georgia_ev_intelligence.runtime_pipeline.schemas import (
    ParentContext,
    RetrievedChildChunk,
)


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


class FailingRetrievalPipeline:
    def retrieve_with_sources(self, question: str) -> HybridRetrievalResult:
        raise RuntimeError(f"boom for {question}")


def test_runner_populates_retrieval_contexts() -> None:
    runner = RetrievalOnlyRunner(retrieval_pipeline_factory=FakeRetrievalPipeline)

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


def test_loads_human_validated_question_workbook(tmp_path) -> None:
    input_path = tmp_path / "human_validated.xlsx"
    pd.DataFrame([
        {
            "Num": 7,
            "Use Case Category": "category",
            "Question": "Which suppliers are listed?",
            "Human validated answers": "Validated answer",
        }
    ]).to_excel(input_path, sheet_name="Sheet1", index=False)

    rows = _load_questions(input_path=input_path, sheet_name="Sheet1")

    assert rows == [
        QuestionRow(
            serial_number=7,
            question="Which suppliers are listed?",
            golden_answer="Validated answer",
        )
    ]


def test_writer_creates_parent_directory_and_workbook(tmp_path) -> None:
    output_path = tmp_path / "nested" / "retrieval_only.xlsx"
    row = RetrievalOnlyRunner(
        retrieval_pipeline_factory=FakeRetrievalPipeline,
    ).run([QuestionRow(serial_number=1, question="q", golden_answer="a")])[0]
    writer = RetrievalWorkbookWriter(output_path=output_path)

    writer.append(row)

    dataframe = pd.read_excel(output_path)
    assert list(dataframe.columns) == OUTPUT_COLUMNS
    assert dataframe.loc[0, "retrieved context"] == "final parent context for q"
    assert dataframe.loc[0, "dense retrieved context"].find("Dense Co") >= 0
    assert dataframe.loc[0, "sparse retrieved context"].find("Sparse Co") >= 0


def test_runner_records_retrieval_errors() -> None:
    runner = RetrievalOnlyRunner(retrieval_pipeline_factory=FailingRetrievalPipeline)

    rows = runner.run([QuestionRow(serial_number=1, question="q", golden_answer="a")])

    assert rows[0].retrieved_context == "ERROR: retrieval failed: boom for q"
    assert rows[0].dense_retrieved_context == "ERROR: retrieval failed: boom for q"
    assert rows[0].sparse_retrieved_context == "ERROR: retrieval failed: boom for q"
