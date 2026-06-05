"""Tests for the cumulative XLSX debug-trace writer."""
from __future__ import annotations

import pandas as pd

from georgia_ev_intelligence.runtime_pipeline.debug_trace import DebugRecorder
from georgia_ev_intelligence.runtime_pipeline.debug_trace.writer import (
    QUESTION_COLUMNS,
    STEP_COLUMNS,
    DebugTraceWriter,
)


def _recorder(query: str) -> DebugRecorder:
    rec = DebugRecorder(original_query=query)
    rec.record("retrieval", summary="ok", details={"n": 1})
    rec.record("generation", status="ok")
    rec.finalize(_chat(query))
    return rec


class _chat:
    def __init__(self, query: str) -> None:
        self.answer = "ans " + query
        self.parent_contexts = [object()]
        self.trace = {"final_outcome": "success", "llm_calls": 2, "attempts": [{}]}
        self.error = ""
        self.warn = ""
        self.effective_query = query
        self.history_used = False
        self.confidence = 3.0


def test_flush_creates_workbook_with_two_sheets(tmp_path) -> None:
    path = tmp_path / "nested" / "trace.xlsx"
    writer = DebugTraceWriter(path)
    writer.flush(_recorder("first question"))

    assert path.exists()
    book = pd.read_excel(path, sheet_name=None)
    assert set(book.keys()) == {"steps", "questions"}
    assert list(book["steps"].columns) == STEP_COLUMNS
    assert list(book["questions"].columns) == QUESTION_COLUMNS
    assert len(book["steps"]) == 2  # two recorded steps
    assert len(book["questions"]) == 1
    assert int(book["steps"]["question_seq"].iloc[0]) == 1


def test_question_seq_increments_across_flushes(tmp_path) -> None:
    path = tmp_path / "trace.xlsx"
    writer = DebugTraceWriter(path)
    writer.flush(_recorder("q1"))
    writer.flush(_recorder("q2"))

    book = pd.read_excel(path, sheet_name=None)
    assert len(book["questions"]) == 2
    assert sorted(book["questions"]["question_seq"].tolist()) == [1, 2]
    # steps from both questions accumulate (2 + 2).
    assert len(book["steps"]) == 4
    assert sorted(book["steps"]["question_seq"].unique().tolist()) == [1, 2]


def test_seq_continues_from_existing_file(tmp_path) -> None:
    path = tmp_path / "trace.xlsx"
    DebugTraceWriter(path).flush(_recorder("q1"))
    # A brand-new writer instance (e.g. process restart) must not reuse seq 1.
    DebugTraceWriter(path).flush(_recorder("q2"))

    book = pd.read_excel(path, sheet_name=None)
    assert sorted(book["questions"]["question_seq"].tolist()) == [1, 2]


def test_oversized_cell_is_truncated(tmp_path) -> None:
    path = tmp_path / "trace.xlsx"
    writer = DebugTraceWriter(path, max_cell_chars=100)
    rec = DebugRecorder(original_query="big")
    rec.record("generation", details={"prompt": "x" * 5000})
    rec.finalize(_chat("big"))
    writer.flush(rec)

    book = pd.read_excel(path, sheet_name=None)
    cell = book["steps"]["details_json"].iloc[0]
    assert len(cell) <= 100 + len(" …[truncated 99999 chars]") + 10
    assert "truncated" in cell
