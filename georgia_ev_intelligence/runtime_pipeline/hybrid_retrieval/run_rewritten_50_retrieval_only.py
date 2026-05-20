"""Run only retrieval over the human-validated QA workbook and export traces."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from georgia_ev_intelligence.runtime_pipeline.schemas import RetrievedChildChunk

from .factory import build_default_pipeline
from .run_rewritten_50 import (
    DEFAULT_OUTPUT_DIR_NAME,
    DEFAULT_QUESTIONS_SHEET,
    DEFAULT_QUESTIONS_WORKBOOK,
    QuestionRow,
    _default_input_path,
    _empty_trace_values,
    _format_retrieved_context,
    _load_questions,
    _project_root,
    _trace_values,
)


TRACE_COLUMNS = [
    "sparse_child_count",
    "dense_child_count",
    "merged_child_result_count",
    "unique_child_chunk_count",
    "unique_parent_id_count",
    "parent_context_count_before_rerank",
    "parent_context_count_after_rerank",
]

OUTPUT_COLUMNS = [
    "s.no",
    "question",
    "human validated answer",
    "retrieved context",
    *TRACE_COLUMNS,
    "dense retrieved context",
    "sparse retrieved context",
]


@dataclass(frozen=True)
class RetrievalOnlyRow:
    """Workbook row containing retrieval traces for one question."""

    serial_number: object
    question: str
    golden_answer: str
    retrieved_context: str
    dense_retrieved_context: str
    sparse_retrieved_context: str
    trace_values: dict[str, object]


class RetrievalOnlyRunner:
    """Run the hybrid retrieval pipeline without answer generation."""

    def __init__(self, retrieval_pipeline_factory=build_default_pipeline) -> None:
        self._retrieval_pipeline_factory = retrieval_pipeline_factory
        self._retrieval_pipeline = None
        self._retrieval_load_error = ""

    def run(self, questions: list[QuestionRow]) -> list[RetrievalOnlyRow]:
        return list(self.iter_rows(questions))

    def iter_rows(self, questions: list[QuestionRow]):
        for index, question_row in enumerate(questions, start=1):
            print(f"[{index}/{len(questions)}] {question_row.question}")
            yield self._retrieve(question_row)

    def _retrieve(self, question_row: QuestionRow) -> RetrievalOnlyRow:
        try:
            retrieval_pipeline = self._get_retrieval_pipeline()
            if hasattr(retrieval_pipeline, "retrieve_with_sources"):
                retrieval_result = retrieval_pipeline.retrieve_with_sources(
                    question_row.question,
                )
                return RetrievalOnlyRow(
                    serial_number=question_row.serial_number,
                    question=question_row.question,
                    golden_answer=question_row.golden_answer,
                    retrieved_context=_format_retrieved_context(
                        retrieval_result.parent_contexts,
                    ),
                    dense_retrieved_context=_format_child_contexts(
                        retrieval_result.dense_children,
                    ),
                    sparse_retrieved_context=_format_child_contexts(
                        retrieval_result.sparse_children,
                    ),
                    trace_values=_trace_values(retrieval_result.trace),
                )

            parent_contexts = retrieval_pipeline.retrieve(question_row.question)
            return RetrievalOnlyRow(
                serial_number=question_row.serial_number,
                question=question_row.question,
                golden_answer=question_row.golden_answer,
                retrieved_context=_format_retrieved_context(parent_contexts),
                dense_retrieved_context="",
                sparse_retrieved_context="",
                trace_values=_empty_trace_values(),
            )
        except Exception as exc:
            error = f"ERROR: retrieval failed: {exc}"
            return RetrievalOnlyRow(
                serial_number=question_row.serial_number,
                question=question_row.question,
                golden_answer=question_row.golden_answer,
                retrieved_context=error,
                dense_retrieved_context=error,
                sparse_retrieved_context=error,
                trace_values=_empty_trace_values(),
            )

    def _get_retrieval_pipeline(self):
        if self._retrieval_pipeline is not None:
            return self._retrieval_pipeline
        if self._retrieval_load_error:
            raise RuntimeError(self._retrieval_load_error)
        try:
            self._retrieval_pipeline = self._retrieval_pipeline_factory()
        except Exception as exc:
            self._retrieval_load_error = str(exc)
            raise
        return self._retrieval_pipeline


class RetrievalWorkbookWriter:
    """Write retrieval-only rows to an XLSX workbook."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self._rows: list[dict[str, object]] = []

    def append(self, row: RetrievalOnlyRow) -> None:
        self._rows.append({
            "s.no": row.serial_number,
            "question": row.question,
            "human validated answer": row.golden_answer,
            "retrieved context": row.retrieved_context,
            **row.trace_values,
            "dense retrieved context": row.dense_retrieved_context,
            "sparse retrieved context": row.sparse_retrieved_context,
        })
        self.write()

    def write(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe = pd.DataFrame(self._rows, columns=OUTPUT_COLUMNS)
        dataframe.to_excel(self._output_path, index=False)


def main() -> int:
    args = _parse_args()
    input_path = args.input or _default_input_path()
    output_path = args.output or _default_output_path()

    questions = _load_questions(input_path=input_path, sheet_name=args.sheet)
    if args.limit is not None:
        questions = questions[: args.limit]

    runner = RetrievalOnlyRunner()
    writer = RetrievalWorkbookWriter(output_path=output_path)
    for row in runner.iter_rows(questions):
        writer.append(row)

    print(f"Saved {len(questions)} retrieval rows to {output_path}")
    return 0


def _format_child_contexts(children: list[RetrievedChildChunk]) -> str:
    sections: list[str] = []
    for index, child in enumerate(children, start=1):
        lines = [
            f"[{index}] chunk_id: {child.chunk_id}",
            f"parent_record_id: {child.parent_record_id}",
            f"chunk_type: {child.chunk_type}",
        ]
        for field_name, value in child.metadata.items():
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            lines.append(f"{field_name}: {text}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        _project_root()
        / "georgia_ev_intelligence"
        / "outputs"
        / DEFAULT_OUTPUT_DIR_NAME
        / f"{timestamp}_retrieval_only.xlsx"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Run only hybrid retrieval on kb/{DEFAULT_QUESTIONS_WORKBOOK} "
            "and write retrieved contexts to XLSX."
        )
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_QUESTIONS_SHEET,
        help=f"Worksheet name inside kb/{DEFAULT_QUESTIONS_WORKBOOK}.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional QA workbook path. Defaults to the human-validated QA workbook.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output XLSX path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke testing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
