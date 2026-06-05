"""Shared pytest fixtures.

Per-question debug tracing (DEBUG_TRACE_ENABLED) is on by default for the live
UI, but it must never write files during the test suite — disable it globally so
exercising QueryDispatcher.dispatch (e.g. test_rag_chat_history) doesn't create a
stray outputs/debug_traces workbook. Tests that specifically target the tracer
opt back in by binding a recorder/writer directly.
"""
from __future__ import annotations

import pytest

from georgia_ev_intelligence.shared.config import settings


@pytest.fixture(autouse=True)
def _disable_debug_trace(monkeypatch):
    monkeypatch.setattr(settings, "DEBUG_TRACE_ENABLED", False, raising=False)
