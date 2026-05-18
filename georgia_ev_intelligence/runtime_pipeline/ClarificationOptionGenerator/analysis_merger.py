"""Merge original and clarification query analyses with phrase classification."""
from __future__ import annotations

import re
from typing import Any

from ..query_analyzer.models import QueryAnalysisResult, VocabularyMatch
from ..phrase_classifier.models import PhraseClassificationResult
from .models import ResolvedClarification, ResolvedQueryContext

_HYPHEN_PATTERN = re.compile(r"(?<=[a-z0-9])-(?=[a-z0-9])", re.IGNORECASE)


def _normalize_term(term: str) -> str:
    text = term.strip().lower()
    text = _HYPHEN_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


class AnalysisMerger:
    """Merge vocabulary matches and classification state after clarification."""

    def merge(
        self,
        original_query: str,
        original_analysis: Any,
        original_phrase_classification: Any,
        resolved_clarifications: list[ResolvedClarification],
        clarification_analysis: Any | None,
        clarification_phrase_classification: Any | None,
    ) -> ResolvedQueryContext:
        merged_vocab = _merge_vocabulary(
            _extract_matches(original_analysis),
            _extract_matches(clarification_analysis),
        )

        operation = _extract_operation(original_analysis)
        target_entity = _extract_target_entity(original_analysis)

        # Use target_entity_override from phrase classification if present.
        override = _extract_target_entity_override(original_phrase_classification)
        if override:
            target_entity = override
        if not target_entity and clarification_phrase_classification is not None:
            clar_override = _extract_target_entity_override(clarification_phrase_classification)
            if clar_override:
                target_entity = clar_override

        # Merge classified term buckets.
        semantic_intent = _merge_list_field(
            original_phrase_classification, clarification_phrase_classification,
            "semantic_intent_terms",
        )
        context_terms = _merge_list_field(
            original_phrase_classification, clarification_phrase_classification,
            "context_terms",
        )
        domain_signal = _merge_list_field(
            original_phrase_classification, clarification_phrase_classification,
            "domain_signal_terms",
        )
        connector_terms = _merge_list_field(
            original_phrase_classification, clarification_phrase_classification,
            "connector_terms",
        )

        # Remove clarified terms from remaining ambiguous.
        resolved_terms = {_normalize_term(rc.term) for rc in resolved_clarifications}
        remaining: list[str] = []
        for term in _extract_ambiguous(original_analysis):
            if _normalize_term(term) in resolved_terms:
                continue
            remaining.append(term)
        if clarification_analysis is not None:
            for term in _extract_ambiguous(clarification_analysis):
                if _normalize_term(term) in resolved_terms:
                    continue
                if term not in remaining:
                    remaining.append(term)

        notes = _build_generation_notes(resolved_clarifications)

        return ResolvedQueryContext(
            original_query=original_query,
            original_analysis=original_analysis,
            original_phrase_classification=original_phrase_classification,
            resolved_clarifications=resolved_clarifications,
            clarification_analysis=clarification_analysis,
            clarification_phrase_classification=clarification_phrase_classification,
            merged_matched_vocabulary=merged_vocab,
            target_entity=target_entity,
            operation=operation,
            semantic_intent_terms=semantic_intent,
            context_terms=context_terms,
            domain_signal_terms=domain_signal,
            connector_terms=connector_terms,
            remaining_ambiguous_terms=remaining,
            final_generation_notes=notes,
            debug={
                "merged_vocabulary_count": len(merged_vocab),
                "remaining_ambiguous_count": len(remaining),
            },
        )


# --- Helpers ---


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


def _extract_operation(analysis: Any) -> str | None:
    if analysis is None:
        return None
    return getattr(analysis, "operation", None)


def _extract_target_entity(analysis: Any) -> str | None:
    if analysis is None:
        return None
    return getattr(analysis, "target_entity", None)


def _extract_target_entity_override(classification: Any) -> str | None:
    if classification is None:
        return None
    if isinstance(classification, PhraseClassificationResult):
        return classification.target_entity_override
    return getattr(classification, "target_entity_override", None)


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


def _merge_list_field(
    original_classification: Any,
    clarification_classification: Any,
    field_name: str,
) -> list[str]:
    """Merge a list field from two PhraseClassificationResults."""
    result: list[str] = []
    seen: set[str] = set()
    for cls in (original_classification, clarification_classification):
        if cls is None:
            continue
        values = getattr(cls, field_name, None)
        if not values:
            continue
        for v in values:
            lower = v.lower().strip()
            if lower not in seen:
                seen.add(lower)
                result.append(v)
    return result


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
