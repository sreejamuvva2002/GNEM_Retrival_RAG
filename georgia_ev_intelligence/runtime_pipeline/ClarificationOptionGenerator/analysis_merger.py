"""Merge original and clarification query analyses."""
from __future__ import annotations

from typing import Any

from ..query_analyzer.models import QueryAnalysisResult, VocabularyMatch
from .concept_registry import normalize_term
from .models import ResolvedClarification, ResolvedQueryContext


class AnalysisMerger:
    """Merge vocabulary matches and ambiguity state after clarification."""

    def merge(
        self,
        original_query: str,
        original_analysis: Any,
        resolved_clarifications: list[ResolvedClarification],
        clarification_analysis: Any | None,
        clarification_id: str,
    ) -> ResolvedQueryContext:
        merged_vocab = _merge_vocabulary(
            _extract_matches(original_analysis),
            _extract_matches(clarification_analysis),
        )

        resolved_terms = {
            normalize_term(rc.term) for rc in resolved_clarifications
        }
        ignored_terms = [
            rc.term
            for rc in resolved_clarifications
            if rc.ignored
        ]
        ignored_normalized = {normalize_term(t) for t in ignored_terms}

        remaining: list[str] = []
        for term in _extract_ambiguous(original_analysis):
            norm = normalize_term(term)
            if norm in resolved_terms or norm in ignored_normalized:
                continue
            remaining.append(term)

        if clarification_analysis is not None:
            for term in _extract_ambiguous(clarification_analysis):
                norm = normalize_term(term)
                if norm in resolved_terms or norm in ignored_normalized:
                    continue
                if term not in remaining:
                    remaining.append(term)

        notes = _build_generation_notes(resolved_clarifications)

        return ResolvedQueryContext(
            clarification_id=clarification_id,
            original_query=original_query,
            original_analysis=original_analysis,
            resolved_clarifications=resolved_clarifications,
            clarification_analysis=clarification_analysis,
            merged_matched_vocabulary=merged_vocab,
            remaining_ambiguous_terms=remaining,
            ignored_ambiguous_terms=ignored_terms,
            final_generation_notes=notes,
            debug={
                "merged_vocabulary_count": len(merged_vocab),
                "remaining_ambiguous_count": len(remaining),
            },
        )


def _extract_matches(analysis: Any) -> list[Any]:
    if analysis is None:
        return []
    if isinstance(analysis, QueryAnalysisResult):
        return list(analysis.matched_vocabulary)
    matches = getattr(analysis, "matched_vocabulary", None)
    return list(matches) if matches else []


def _extract_ambiguous(analysis: Any) -> list[str]:
    if analysis is None:
        return []
    if isinstance(analysis, QueryAnalysisResult):
        return list(analysis.ambiguous_terms)
    terms = getattr(analysis, "ambiguous_terms", None)
    return list(terms) if terms else []


def _vocab_key(match: Any) -> tuple[str, str, str | None]:
    if isinstance(match, VocabularyMatch):
        return (match.canonical_value, match.source_column, match.term_type)
    return (
        str(getattr(match, "canonical_value", "")),
        str(getattr(match, "source_column", "")),
        getattr(match, "term_type", None),
    )


def _merge_vocabulary(
    original_matches: list[Any],
    clarification_matches: list[Any],
) -> list[Any]:
    merged: dict[tuple[str, str, str | None], Any] = {}
    for match in original_matches + clarification_matches:
        merged[_vocab_key(match)] = match
    return list(merged.values())


def _build_generation_notes(
    resolved_clarifications: list[ResolvedClarification],
) -> list[str]:
    notes: list[str] = []
    for rc in resolved_clarifications:
        if rc.ignored:
            notes.append(f"The user asked to ignore the phrase '{rc.term}'.")
        elif rc.meaning:
            notes.append(
                f"The user clarified that '{rc.term}' means: {rc.meaning}."
            )
    return notes
