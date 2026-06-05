"""Cumulative XLSX writer for per-question debug traces.

Maintains a single workbook with two sheets:

- ``steps``     — one row per pipeline step (the detailed debug log)
- ``questions`` — one summary row per question

Each ``flush`` reads the existing workbook (if any), appends the recorder's
rows under a freshly-allocated ``question_seq``, and rewrites both sheets. A
process-wide lock serialises concurrent flushes; an in-process counter avoids
re-reading the whole file just to find the next sequence number.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, List, Optional

from ...shared import config
from .recorder import DebugRecorder

# Stable column order for each sheet.
STEP_COLUMNS = [
    "session_id",
    "question_seq",
    "step_index",
    "wall_time",
    "elapsed_ms",
    "duration_ms",
    "original_query",
    "effective_query",
    "step",
    "attempt",
    "sub_step",
    "status",
    "summary",
    "error",
    "details_json",
]

QUESTION_COLUMNS = [
    "session_id",
    "question_seq",
    "started_at",
    "ended_at",
    "original_query",
    "effective_query",
    "history_used",
    "self_healing_enabled",
    "final_outcome",
    "attempts",
    "total_llm_calls",
    "num_parent_contexts",
    "answer_chars",
    "confidence",
    "warn",
    "error",
    "answer_preview",
]


class DebugTraceWriter:
    """Append-only (read-modify-rewrite) writer for the cumulative debug workbook."""

    def __init__(self, path: str | Path, max_cell_chars: int = 32000) -> None:
        self._path = Path(path)
        self._max_cell_chars = max_cell_chars
        self._lock = threading.Lock()
        self._next_seq: Optional[int] = None

    @property
    def path(self) -> Path:
        return self._path

    def flush(self, recorder: DebugRecorder) -> None:
        import pandas as pd

        from datetime import datetime

        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)

            existing_steps, existing_questions, max_seq = self._read_existing(pd)
            seq = self._allocate_seq(max_seq)

            step_rows = [
                {**row, "question_seq": seq} for row in recorder.rows
            ]
            question_row = self._build_question_row(recorder, seq, datetime.now())

            steps_df = self._cap_frame(self._frame(pd, existing_steps, step_rows, STEP_COLUMNS))
            questions_df = self._cap_frame(
                self._frame(pd, existing_questions, [question_row], QUESTION_COLUMNS)
            )

            with pd.ExcelWriter(self._path, engine="openpyxl") as xl:
                steps_df.to_excel(xl, sheet_name="steps", index=False)
                questions_df.to_excel(xl, sheet_name="questions", index=False)

    # -- internals -------------------------------------------------------- #
    def _read_existing(self, pd: Any):
        """Return (steps_df_or_None, questions_df_or_None, max_existing_seq)."""
        if not self._path.exists():
            return None, None, 0
        try:
            book = pd.read_excel(self._path, sheet_name=None)
        except Exception:
            # Corrupt / unreadable file — start fresh rather than crash a request.
            return None, None, 0
        steps = book.get("steps")
        questions = book.get("questions")
        max_seq = 0
        for frame in (steps, questions):
            if frame is not None and "question_seq" in frame.columns and len(frame):
                try:
                    max_seq = max(max_seq, int(frame["question_seq"].max()))
                except (ValueError, TypeError):
                    pass
        return steps, questions, max_seq

    def _allocate_seq(self, max_existing: int) -> int:
        if self._next_seq is None:
            self._next_seq = max_existing + 1
        else:
            self._next_seq = max(self._next_seq, max_existing + 1)
        seq = self._next_seq
        self._next_seq += 1
        return seq

    @staticmethod
    def _build_question_row(recorder: DebugRecorder, seq: int, ended) -> dict:
        s = recorder.summary or {}
        return {
            "session_id": recorder.session_id,
            "question_seq": seq,
            "started_at": recorder.started_wall,
            "ended_at": ended.isoformat(timespec="seconds"),
            "original_query": recorder.original_query,
            "effective_query": s.get("effective_query", recorder.effective_query),
            "history_used": s.get("history_used", ""),
            "self_healing_enabled": config.SELF_HEALING_ENABLED,
            "final_outcome": s.get("final_outcome", ""),
            "attempts": s.get("attempts", ""),
            "total_llm_calls": s.get("total_llm_calls", ""),
            "num_parent_contexts": s.get("num_parent_contexts", ""),
            "answer_chars": s.get("answer_chars", ""),
            "confidence": s.get("confidence", ""),
            "warn": s.get("warn", ""),
            "error": s.get("error", ""),
            "answer_preview": s.get("answer_preview", ""),
        }

    @staticmethod
    def _frame(pd: Any, existing, new_rows: List[dict], columns: List[str]):
        new_df = pd.DataFrame(new_rows, columns=columns)
        if existing is None or len(existing) == 0:
            return new_df
        combined = pd.concat([existing, new_df], ignore_index=True)
        # Keep a stable, known column order even if the on-disk file is older.
        ordered = [c for c in columns if c in combined.columns]
        extra = [c for c in combined.columns if c not in columns]
        return combined[ordered + extra]

    def _cap_frame(self, frame: Any) -> Any:
        # DataFrame.map (pandas >= 2.1) supersedes the deprecated applymap.
        mapper = getattr(frame, "map", None) or frame.applymap
        return mapper(self._cap_cell)

    def _cap_cell(self, value: Any) -> Any:
        if isinstance(value, str) and len(value) > self._max_cell_chars:
            dropped = len(value) - self._max_cell_chars
            return value[: self._max_cell_chars] + f" …[truncated {dropped} chars]"
        return value


# --------------------------------------------------------------------------- #
# Process-wide default writer (the cumulative UI workbook)
# --------------------------------------------------------------------------- #
_DEFAULT: Optional[DebugTraceWriter] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_writer() -> DebugTraceWriter:
    """Singleton writer for the cumulative UI debug workbook (settings-driven)."""
    global _DEFAULT
    if _DEFAULT is None:
        with _DEFAULT_LOCK:
            if _DEFAULT is None:
                _DEFAULT = DebugTraceWriter(
                    config.DEBUG_TRACE_PATH,
                    max_cell_chars=config.DEBUG_TRACE_MAX_CELL_CHARS,
                )
    return _DEFAULT
