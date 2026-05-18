"""Data models for open-ended clarification generation and resolution."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ClarificationQuestion:
    """An open-ended clarification question for one ambiguous phrase."""

    term: str
    question: str


@dataclass
class ClarificationRequest:
    """Request containing open-ended clarification questions."""

    clarification_required: bool
    clarification_id: str
    original_query: str
    questions: list[ClarificationQuestion]
    status: str = "pending"
    created_at: datetime | None = None


@dataclass(frozen=True)
class ClarificationAnswer:
    """User's free-text clarification for one term."""

    term: str
    custom_text: str


@dataclass(frozen=True)
class ClarificationSubmission:
    """Collection of user answers for a clarification request."""

    clarification_id: str
    answers: list[ClarificationAnswer]


@dataclass(frozen=True)
class ResolvedClarification:
    """A resolved clarification for one term."""

    term: str
    meaning: str
    ignored: bool = False


@dataclass
class StoredClarificationSession:
    """Persisted state of a clarification session."""

    clarification_id: str
    original_query: str
    original_analysis: Any
    phrase_classification: Any
    request: ClarificationRequest
    status: str
    resolved_context: Any | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class ResolvedQueryContext:
    """Complete resolved context after analysis, classification, and optional clarification."""

    original_query: str
    original_analysis: Any
    original_phrase_classification: Any
    resolved_clarifications: list[ResolvedClarification] = field(default_factory=list)
    clarification_analysis: Any | None = None
    clarification_phrase_classification: Any | None = None
    merged_matched_vocabulary: list[Any] = field(default_factory=list)
    target_entity: str | None = None
    operation: str | None = None
    semantic_intent_terms: list[str] = field(default_factory=list)
    context_terms: list[str] = field(default_factory=list)
    domain_signal_terms: list[str] = field(default_factory=list)
    connector_terms: list[str] = field(default_factory=list)
    remaining_ambiguous_terms: list[str] = field(default_factory=list)
    final_generation_notes: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
