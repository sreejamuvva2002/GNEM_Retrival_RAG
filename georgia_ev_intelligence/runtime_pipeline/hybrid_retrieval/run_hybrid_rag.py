"""Shared utility helpers for pipeline runners.

This module exposes constants, dataclasses, and helper functions that are
imported by the top-level runner (``run_baseline.py``) and the evaluation
scripts.  It does **not** contain a ``main()`` entry point — use
``run_baseline.py`` to launch pipeline runs.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext


# ---------------------------------------------------------------------------
# Default workbook / sheet
# ---------------------------------------------------------------------------

DEFAULT_QUESTIONS_WORKBOOK = "Rewritten_queries.xlsx"
DEFAULT_QUESTIONS_SHEET = "Sheet1"


# ---------------------------------------------------------------------------
# Column-name candidates (supports both old and new file conventions)
# ---------------------------------------------------------------------------

QUESTION_COLUMN_CANDIDATES = (
    "question",
    "Question",
    "Original Question",
)

GOLDEN_ANSWER_COLUMN_CANDIDATES = (
    "golden_answer",
    "Golden Answer",
    "answer",
    "Answer",
    "human_validated_answer",
    "Human Validated Answer",
    "Human validated answers",
    "validated_answer",
)

# Optional rewritten-query columns produced by multi-query generation.
# Each non-empty value is an alternative phrasing of the original question.
# Supports both naming conventions: "rewritten_query_N" and "Variation N".
REWRITTEN_QUERY_COLUMNS = (
    "rewritten_query_1",
    "rewritten_query_2",
    "rewritten_query_3",
    "rewritten_query_4",
    "rewritten_query_5",
    "Variation 1",
    "Variation 2",
    "Variation 3",
    "Variation 4",
    "Variation 5",
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuestionRow:
    serial_number: object
    question: str
    golden_answer: str
    rewritten_queries: tuple[str, ...] = ()
    """Additional query variants for multi-query retrieval (may be empty)."""


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------

def _load_questions(input_path: Path, sheet_name: str) -> list[QuestionRow]:
    dataframe = _read_sheet_with_fallback(input_path, sheet_name)
    column_map = _question_column_map(dataframe, input_path)

    rows: list[QuestionRow] = []
    for record in dataframe.to_dict(orient="records"):
        question = str(record[column_map["question"]]).strip()
        if not question:
            continue

        rewrites: list[str] = []
        for col in REWRITTEN_QUERY_COLUMNS:
            if col in dataframe.columns:
                val = record.get(col)
                if val is not None and not pd.isna(val):
                    stripped = str(val).strip()
                    if stripped:
                        rewrites.append(stripped)

        rows.append(QuestionRow(
            serial_number=record[column_map["serial_number"]],
            question=question,
            golden_answer=(
                ""
                if pd.isna(record[column_map["answer"]])
                else str(record[column_map["answer"]])
            ),
            rewritten_queries=tuple(rewrites),
        ))

    return rows


def _read_sheet_with_fallback(input_path: Path, sheet_name: str) -> pd.DataFrame:
    """Read the named sheet; fall back to the first sheet with a warning."""
    xl = pd.ExcelFile(input_path)
    if sheet_name in xl.sheet_names:
        return pd.read_excel(xl, sheet_name=sheet_name)
    warnings.warn(
        f"Sheet '{sheet_name}' not found in {input_path.name}. "
        f"Available sheets: {xl.sheet_names}. "
        f"Falling back to first sheet: '{xl.sheet_names[0]}'.",
        stacklevel=3,
    )
    return pd.read_excel(xl, sheet_name=0)


def _question_column_map(dataframe: pd.DataFrame, input_path: Path) -> dict[str, str]:
    """Return canonical question columns for supported QA workbooks."""
    question_column = _first_existing_column(
        dataframe,
        QUESTION_COLUMN_CANDIDATES,
        input_path,
        "question",
    )
    answer_column = _first_existing_column(
        dataframe,
        GOLDEN_ANSWER_COLUMN_CANDIDATES,
        input_path,
        "golden answer",
    )
    serial_column = _optional_first_existing_column(dataframe, ("s.no", "Num"))
    return {
        "serial_number": serial_column or question_column,
        "question": question_column,
        "answer": answer_column,
    }


def _first_existing_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    input_path: Path,
    label: str,
) -> str:
    column = _optional_first_existing_column(dataframe, candidates)
    if column is not None:
        return column

    raise ValueError(
        f"{input_path} is missing a {label} column. "
        f"Supported {label} columns: {', '.join(candidates)}. "
        f"Found: {', '.join(str(column) for column in dataframe.columns)}"
    )


def _optional_first_existing_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    columns = set(dataframe.columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Retrieval formatting helpers
# ---------------------------------------------------------------------------

def _format_retrieved_context(parent_contexts: list[ParentContext]) -> str:
    return "\n\n".join(
        parent.parent_chunk_text
        for parent in parent_contexts
        if parent.parent_chunk_text
    )


# ---------------------------------------------------------------------------
# Retrieval trace helpers
# ---------------------------------------------------------------------------

def _empty_trace_values() -> dict[str, object]:
    return {
        "sparse_child_count": None,
        "dense_child_count": None,
        "merged_child_result_count": None,
        "unique_child_chunk_count": None,
        "unique_parent_id_count": None,
        "parent_context_count_before_rerank": None,
        "parent_context_count_after_rerank": None,
    }


def _trace_values(trace) -> dict[str, object]:
    if trace is None:
        return _empty_trace_values()
    return {
        "sparse_child_count": trace.sparse_child_count,
        "dense_child_count": trace.dense_child_count,
        "merged_child_result_count": trace.merged_child_result_count,
        "unique_child_chunk_count": trace.unique_child_chunk_count,
        "unique_parent_id_count": trace.unique_parent_id_count,
        "parent_context_count_before_rerank": trace.parent_context_count_before_rerank,
        "parent_context_count_after_rerank": trace.parent_context_count_after_rerank,
    }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _default_input_path() -> Path:
    return _project_root() / "kb" / DEFAULT_QUESTIONS_WORKBOOK


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]
