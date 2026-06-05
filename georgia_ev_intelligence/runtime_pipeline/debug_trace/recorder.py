"""Ambient, context-local debug recorder for per-question step tracing.

A :class:`DebugRecorder` is bound to a :class:`contextvars.ContextVar` for the
duration of one user question (see :func:`session`). Any component anywhere in
the call stack can then call the module-level :func:`record_step` to append a
fully structured row describing what happened at that step — without any
component needing the recorder threaded through its signature.

If no recorder is bound (unit tests, batch scripts, or the UI feature disabled)
:func:`record_step` is a no-op, so instrumentation has zero cost and zero
behavioural impact outside an active UI session.
"""
from __future__ import annotations

import contextlib
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

# The currently-bound recorder for this execution context (thread / async task).
_CURRENT: ContextVar[Optional["DebugRecorder"]] = ContextVar(
    "debug_trace_recorder", default=None
)


@dataclass
class DebugRecorder:
    """Accumulates one row per pipeline step for a single question."""

    original_query: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    effective_query: str = ""
    started_wall: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    rows: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    _start_perf: float = field(default_factory=time.perf_counter)
    _step_index: int = 0

    # -- step capture ----------------------------------------------------- #
    def record(
        self,
        step: str,
        *,
        summary: str = "",
        status: str = "info",
        attempt: Optional[int] = None,
        sub_step: str = "",
        details: Optional[Dict[str, Any]] = None,
        error: str = "",
        duration_ms: Optional[float] = None,
    ) -> None:
        elapsed_ms = round((time.perf_counter() - self._start_perf) * 1000.0, 1)
        self._step_index += 1
        self.rows.append(
            {
                "session_id": self.session_id,
                "step_index": self._step_index,
                "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                "elapsed_ms": elapsed_ms,
                "duration_ms": round(duration_ms, 1) if duration_ms is not None else "",
                "original_query": self.original_query,
                "effective_query": self.effective_query,
                "step": step,
                "attempt": attempt if attempt is not None else "",
                "sub_step": sub_step,
                "status": status,
                "summary": summary,
                "error": error,
                "details_json": _to_json(details) if details else "",
            }
        )

    # -- per-question summary -------------------------------------------- #
    def finalize(self, chat_result: Any) -> None:
        """Capture the per-question summary fields from the returned ChatResult.

        Defensive: works with anything exposing the ChatResult attributes and
        never raises (tracing must not break a request)."""
        try:
            answer = getattr(chat_result, "answer", "") or ""
            trace = getattr(chat_result, "trace", {}) or {}
            healing = trace if isinstance(trace, dict) else {}
            effective = getattr(chat_result, "effective_query", "") or self.effective_query
            if effective:
                self.effective_query = effective
            self.summary = {
                "effective_query": effective,
                "history_used": getattr(chat_result, "history_used", ""),
                "final_outcome": healing.get("final_outcome", ""),
                "attempts": len(healing.get("attempts", []) or []),
                "total_llm_calls": healing.get("llm_calls", ""),
                "num_parent_contexts": len(getattr(chat_result, "parent_contexts", []) or []),
                "answer_chars": len(answer),
                "confidence": _blankable(getattr(chat_result, "confidence", None)),
                "warn": getattr(chat_result, "warn", "") or "",
                "error": getattr(chat_result, "error", "") or "",
                "answer_preview": answer,
            }
        except Exception:
            # Never let summary capture break the response path.
            self.summary = self.summary or {}


# --------------------------------------------------------------------------- #
# Ambient access
# --------------------------------------------------------------------------- #
def current_recorder() -> Optional[DebugRecorder]:
    """The recorder bound to the current context, or ``None`` if tracing is off."""
    return _CURRENT.get()


def record_step(step: str, **kwargs: Any) -> None:
    """Append a step row to the active recorder; no-op when none is bound."""
    recorder = _CURRENT.get()
    if recorder is None:
        return
    try:
        recorder.record(step, **kwargs)
    except Exception:
        # Debug tracing must never break the pipeline it observes.
        pass


def set_effective_query(query: str) -> None:
    """Record the rewritten/standalone query on the active recorder (no-op if off)."""
    recorder = _CURRENT.get()
    if recorder is not None and query:
        recorder.effective_query = query


@contextlib.contextmanager
def session(original_query: str, writer: Any) -> Iterator[DebugRecorder]:
    """Bind a fresh recorder for one question, flushing it on exit.

    The recorder is flushed via ``writer.flush(recorder)`` in a ``finally`` so a
    crashing request still persists whatever steps were recorded. All flush
    errors are swallowed."""
    recorder = DebugRecorder(original_query=original_query)
    token = _CURRENT.set(recorder)
    try:
        yield recorder
    finally:
        _CURRENT.reset(token)
        if writer is not None:
            try:
                writer.flush(recorder)
            except Exception:
                pass


def _blankable(value: Any) -> Any:
    return "" if value is None else value


def _to_json(details: Dict[str, Any]) -> str:
    import json

    try:
        return json.dumps(details, ensure_ascii=False, default=_json_default, indent=2)
    except Exception:
        return repr(details)


def _json_default(obj: Any) -> Any:
    # dataclasses, sets, and other odd objects -> something serializable.
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    return str(obj)
