"""Run retrieval-only over a question list using the web-indexed Neon DB.

Supports three input modes:
  1. --questions "What EV suppliers are in Georgia?" "..."   (inline CLI args)
  2. --input questions.txt            (one question per line, plain text)
  3. --input workbook.xlsx            (uses same workbook format as the
                                       human-validated QA pipeline)

Output is always a timestamped XLSX under
  georgia_ev_intelligence/outputs/web_retrieval/
with columns matching the retrieval-only trace format used by
run_rewritten_50_retrieval_only.py.

Usage examples
--------------
# Inline question(s):
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_web_retrieval \\
    --questions "What EV battery suppliers operate in Georgia?"

# Smoke-test with a workbook (first 5 rows):
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_web_retrieval \\
    --input kb/Human\\ validated\\ 50\\ questions.xlsx --limit 5

# Full workbook run with explicit output path:
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_web_retrieval \\
    --input kb/Rewritten_50_questions.xlsx \\
    --output georgia_ev_intelligence/outputs/web_retrieval/my_run.xlsx
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from georgia_ev_intelligence.runtime_pipeline.schemas import RetrievedChildChunk

from .factory import build_default_pipeline
from .run_rewritten_50 import (
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR_NAME = "web_retrieval"

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
    "retrieved_context",
    *TRACE_COLUMNS,
    "dense_retrieved_context",
    "sparse_retrieved_context",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WebRetrievalRow:
    """One output row: retrieval traces for a single question."""

    serial_number: object
    question: str
    retrieved_context: str
    dense_retrieved_context: str
    sparse_retrieved_context: str
    trace_values: dict[str, object]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class WebRetrievalRunner:
    """Run the hybrid pipeline (BM25 + pgvector → Neon) without LLM generation."""

    def __init__(self, retrieval_pipeline_factory=build_default_pipeline) -> None:
        self._retrieval_pipeline_factory = retrieval_pipeline_factory
        self._retrieval_pipeline = None
        self._retrieval_load_error = ""

    def run(self, questions: list[QuestionRow]) -> list[WebRetrievalRow]:
        return list(self.iter_rows(questions))

    def iter_rows(self, questions: list[QuestionRow]):
        for index, question_row in enumerate(questions, start=1):
            print(f"[{index}/{len(questions)}] {question_row.question}")
            yield self._retrieve(question_row)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(self, question_row: QuestionRow) -> WebRetrievalRow:
        try:
            pipeline = self._get_pipeline()
            if hasattr(pipeline, "retrieve_with_sources"):
                result = pipeline.retrieve_with_sources(question_row.question)
                return WebRetrievalRow(
                    serial_number=question_row.serial_number,
                    question=question_row.question,
                    retrieved_context=_format_retrieved_context(
                        result.parent_contexts,
                    ),
                    dense_retrieved_context=_format_child_contexts(
                        result.dense_children,
                    ),
                    sparse_retrieved_context=_format_child_contexts(
                        result.sparse_children,
                    ),
                    trace_values=_trace_values(result.trace),
                )

            # Fallback: pipeline only exposes .retrieve()
            parent_contexts = pipeline.retrieve(question_row.question)
            return WebRetrievalRow(
                serial_number=question_row.serial_number,
                question=question_row.question,
                retrieved_context=_format_retrieved_context(parent_contexts),
                dense_retrieved_context="",
                sparse_retrieved_context="",
                trace_values=_empty_trace_values(),
            )

        except Exception as exc:
            error = f"ERROR: retrieval failed: {exc}"
            return WebRetrievalRow(
                serial_number=question_row.serial_number,
                question=question_row.question,
                retrieved_context=error,
                dense_retrieved_context=error,
                sparse_retrieved_context=error,
                trace_values=_empty_trace_values(),
            )

    def _get_pipeline(self):
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


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class WebRetrievalWorkbookWriter:
    """Incrementally append rows and flush to XLSX after every question."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self._rows: list[dict[str, object]] = []

    def append(self, row: WebRetrievalRow) -> None:
        self._rows.append({
            "s.no": row.serial_number,
            "question": row.question,
            "retrieved_context": row.retrieved_context,
            **row.trace_values,
            "dense_retrieved_context": row.dense_retrieved_context,
            "sparse_retrieved_context": row.sparse_retrieved_context,
        })
        self.flush()

    def flush(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe = pd.DataFrame(self._rows, columns=OUTPUT_COLUMNS)
        dataframe.to_excel(self._output_path, index=False)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _load_from_text_file(path: Path) -> list[QuestionRow]:
    """Load one question per non-blank line from a plain-text file."""
    rows: list[QuestionRow] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        question = line.strip()
        if question:
            rows.append(QuestionRow(
                serial_number=index,
                question=question,
                golden_answer="",
            ))
    return rows


def _load_from_inline(questions: list[str]) -> list[QuestionRow]:
    """Wrap inline CLI questions as QuestionRow objects."""
    return [
        QuestionRow(serial_number=index, question=q.strip(), golden_answer="")
        for index, q in enumerate(questions, start=1)
        if q.strip()
    ]


def _load_input(args: argparse.Namespace) -> list[QuestionRow]:
    """Resolve the input source and return a list of QuestionRow objects."""
    if args.questions:
        return _load_from_inline(args.questions)

    input_path: Path = args.input or _default_input_path()

    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        return _load_questions(input_path=input_path, sheet_name=args.sheet)

    if input_path.suffix.lower() == ".txt":
        return _load_from_text_file(input_path)

    raise ValueError(
        f"Unsupported input file type: {input_path.suffix!r}. "
        "Expected .xlsx, .xls, or .txt"
    )


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        _project_root()
        / "georgia_ev_intelligence"
        / "outputs"
        / OUTPUT_DIR_NAME
        / f"{timestamp}_web_retrieval.xlsx"
    )


# ---------------------------------------------------------------------------
# Child-context formatter (mirrors run_rewritten_50_retrieval_only.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    output_path = args.output or _default_output_path()

    questions = _load_input(args)
    if args.limit is not None:
        questions = questions[: args.limit]

    if not questions:
        print("No questions to process. Provide --questions or --input.")
        return 1

    runner = WebRetrievalRunner()
    writer = WebRetrievalWorkbookWriter(output_path=output_path)

    for row in runner.iter_rows(questions):
        writer.append(row)

    print(f"Saved {len(questions)} retrieval row(s) → {output_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run hybrid retrieval (BM25 + pgvector/Neon) on a set of questions "
            "and export context traces to XLSX. No LLM generation is performed."
        ),
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--questions",
        nargs="+",
        metavar="QUESTION",
        help="One or more questions passed directly on the command line.",
    )
    input_group.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Path to an input file. Supported formats:\n"
            "  .xlsx / .xls  — QA workbook (same format as the human-validated 50).\n"
            f"                   Defaults to kb/{DEFAULT_QUESTIONS_WORKBOOK}.\n"
            "  .txt          — plain text, one question per line."
        ),
    )

    parser.add_argument(
        "--sheet",
        default=DEFAULT_QUESTIONS_SHEET,
        help=f"Worksheet name when reading an Excel workbook (default: {DEFAULT_QUESTIONS_SHEET!r}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output XLSX path. Defaults to "
            f"georgia_ev_intelligence/outputs/{OUTPUT_DIR_NAME}/<timestamp>_web_retrieval.xlsx."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N questions (useful for smoke testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
