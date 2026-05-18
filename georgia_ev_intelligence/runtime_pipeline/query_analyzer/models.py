"""Data models for deterministic query analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MatchType = Literal["canonical", "alias", "normalized"]


@dataclass(frozen=True)
class VocabularyTerm:
    """A searchable vocabulary entry."""

    canonical_value: str
    normalized_value: str
    source_column: str
    term_type: str | None = None
    id: int | None = None
    aliases: list[str] = field(default_factory=list)
    row_ids: list[int] | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class VocabularyMatch:
    """A vocabulary phrase matched inside a user query."""

    matched_text: str
    canonical_value: str
    source_column: str
    term_type: str | None
    match_type: MatchType
    start_token: int
    end_token: int
    token_length: int = 0
    confidence: float = 1.0
    vocabulary_term_id: int | None = None


@dataclass(frozen=True)
class QueryAnalysisResult:
    """Structured output from QueryAnalyzer.analyze()."""

    original_query: str
    normalized_query: str
    operation: str | None
    target_entity: str | None
    matched_vocabulary: list[VocabularyMatch] = field(default_factory=list)
    ambiguous_terms: list[str] = field(default_factory=list)
    ignored_tokens: list[str] = field(default_factory=list)
    unmatched_tokens: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
