"""
Reasoning layer controller / facade.

Provides a clean public API for intent detection, intent application,
and support-level classification. Consumers import from here instead
of reaching into reasoning/retriever.py internals.
"""
from __future__ import annotations

from .retriever import (
    apply_intent,
    detect_intent,
    support_level,
    RetrievalResult,
)

__all__ = [
    "apply_intent",
    "detect_intent",
    "support_level",
    "RetrievalResult",
]
