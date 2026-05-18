"""Data models for LLM remaining phrase classification."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PhraseCategory(str, Enum):
    """Allowed categories for leftover phrase classification."""

    target_entity = "target_entity"
    context_description = "context_description"
    intent_connector = "intent_connector"
    domain_signal = "domain_signal"
    semantic_intent = "semantic_intent"
    ambiguous_concept = "ambiguous_concept"
    irrelevant_phrase = "irrelevant_phrase"


VALID_CATEGORIES: frozenset[str] = frozenset(c.value for c in PhraseCategory)


@dataclass(frozen=True)
class ClassifiedPhrase:
    """Result of LLM classification for one leftover phrase."""

    phrase: str
    category: PhraseCategory
    needs_clarification: bool
    clarification_question: str | None
    reason: str


@dataclass(frozen=True)
class AmbiguousTerm:
    """A phrase classified as ambiguous_concept that needs user clarification."""

    phrase: str
    clarification_question: str


@dataclass(frozen=True)
class PhraseClassificationResult:
    """Full result of classifying all remaining unmatched phrases."""

    classified_phrases: list[ClassifiedPhrase] = field(default_factory=list)
    clarification_required: bool = False
    ambiguous_terms: list[AmbiguousTerm] = field(default_factory=list)
    target_entity_override: str | None = None
    semantic_intent_terms: list[str] = field(default_factory=list)
    context_terms: list[str] = field(default_factory=list)
    connector_terms: list[str] = field(default_factory=list)
    domain_signal_terms: list[str] = field(default_factory=list)
    irrelevant_terms: list[str] = field(default_factory=list)
    raw_response: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)
