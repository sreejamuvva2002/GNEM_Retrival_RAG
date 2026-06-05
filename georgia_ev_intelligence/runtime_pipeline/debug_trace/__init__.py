"""Per-question debug tracing: detailed, one-row-per-step XLSX logging.

Public API:

- :func:`record_step` — append a structured step row to the active recorder
  (a no-op when no session is bound, e.g. tests / batch scripts / feature off).
- :func:`set_effective_query` — record the rewritten query on the active recorder.
- :func:`session` — context manager that binds a recorder for one question and
  flushes it to the workbook on exit.
- :func:`get_default_writer` — the cumulative UI workbook writer (settings-driven).
- :class:`DebugRecorder`, :class:`DebugTraceWriter` — the underlying types.
"""
from .recorder import (
    DebugRecorder,
    current_recorder,
    record_step,
    session,
    set_effective_query,
)
from .writer import DebugTraceWriter, get_default_writer

__all__ = [
    "DebugRecorder",
    "DebugTraceWriter",
    "current_recorder",
    "get_default_writer",
    "record_step",
    "session",
    "set_effective_query",
]
