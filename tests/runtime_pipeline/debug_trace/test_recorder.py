"""Tests for the ambient debug recorder."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from georgia_ev_intelligence.runtime_pipeline.debug_trace import (
    current_recorder,
    record_step,
    session,
    set_effective_query,
)


class _RecordingWriter:
    def __init__(self) -> None:
        self.flushed = []

    def flush(self, recorder) -> None:
        self.flushed.append(recorder)


@dataclass
class _FakeChatResult:
    answer: str = "There are 2 plants in Georgia."
    parent_contexts: List[object] = field(default_factory=lambda: [object(), object()])
    trace: dict = field(default_factory=lambda: {"final_outcome": "success", "llm_calls": 5,
                                                 "attempts": [{}, {}]})
    error: str = ""
    warn: str = ""
    effective_query: str = "plants in Georgia"
    history_used: bool = True
    confidence: float = 7.5


def test_record_step_is_noop_without_session() -> None:
    # No recorder bound -> nothing happens, no error.
    assert current_recorder() is None
    record_step("retrieval", summary="should be dropped")
    set_effective_query("ignored")
    assert current_recorder() is None


def test_session_collects_rows_and_flushes() -> None:
    writer = _RecordingWriter()
    with session("How many plants?", writer) as rec:
        set_effective_query("plants in Georgia")
        record_step("retrieval", summary="ok", details={"count": 3})
        record_step("generation", status="warn", attempt=1, sub_step="regenerate")
        assert current_recorder() is rec

    # Recorder is unbound again after the context exits.
    assert current_recorder() is None
    # Flushed exactly once with our recorder.
    assert writer.flushed == [rec]
    assert len(rec.rows) == 2

    first = rec.rows[0]
    assert first["step"] == "retrieval"
    assert first["effective_query"] == "plants in Georgia"
    assert first["step_index"] == 1
    assert first["original_query"] == "How many plants?"
    assert '"count": 3' in first["details_json"]

    second = rec.rows[1]
    assert second["step"] == "generation"
    assert second["status"] == "warn"
    assert second["attempt"] == 1
    assert second["sub_step"] == "regenerate"
    assert second["step_index"] == 2


def test_finalize_captures_summary_from_chat_result() -> None:
    writer = _RecordingWriter()
    with session("q", writer) as rec:
        rec.finalize(_FakeChatResult())

    s = rec.summary
    assert s["final_outcome"] == "success"
    assert s["attempts"] == 2
    assert s["total_llm_calls"] == 5
    assert s["num_parent_contexts"] == 2
    assert s["history_used"] is True
    assert s["confidence"] == 7.5
    assert s["answer_chars"] == len("There are 2 plants in Georgia.")


def test_session_flushes_even_on_exception() -> None:
    writer = _RecordingWriter()
    try:
        with session("q", writer):
            record_step("retrieval")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert len(writer.flushed) == 1
    assert current_recorder() is None
