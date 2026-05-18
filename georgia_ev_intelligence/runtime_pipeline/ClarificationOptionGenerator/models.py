"""Data models for clarification option generation and resolution."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ClarificationOption:
    id: str
    label: str
    meaning: str
    requires_custom_text: bool = False
    ignore_term: bool = False


@dataclass(frozen=True)
class ConceptDefinition:
    concept_id: str
    aliases: list[str]
    description: str
    clarification_options: list[ClarificationOption]


@dataclass(frozen=True)
class AmbiguousTermClarification:
    term: str
    normalized_term: str
    matched_concept_id: str | None
    question: str
    options: list[ClarificationOption]


@dataclass
class ClarificationRequest:
    clarification_required: bool
    clarification_id: str
    original_query: str
    ambiguous_terms: list[AmbiguousTermClarification]
    status: str = "pending"
    created_at: datetime | None = None


@dataclass(frozen=True)
class ClarificationAnswer:
    term: str
    selected_option_id: str | None = None
    custom_text: str | None = None


@dataclass(frozen=True)
class ClarificationSubmission:
    clarification_id: str
    answers: list[ClarificationAnswer]


@dataclass(frozen=True)
class ResolvedClarification:
    term: str
    selected_option_id: str | None
    meaning: str | None
    ignored: bool
    custom_text: str | None
    matched_concept_id: str | None = None


@dataclass
class ResolvedQueryContext:
    clarification_id: str
    original_query: str
    original_analysis: Any
    resolved_clarifications: list[ResolvedClarification] = field(default_factory=list)
    clarification_analysis: Any | None = None
    merged_matched_vocabulary: list[Any] = field(default_factory=list)
    remaining_ambiguous_terms: list[str] = field(default_factory=list)
    ignored_ambiguous_terms: list[str] = field(default_factory=list)
    final_generation_notes: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class StoredClarificationSession:
    clarification_id: str
    original_query: str
    original_analysis: Any
    request: ClarificationRequest
    status: str
    resolved_context: ResolvedQueryContext | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
