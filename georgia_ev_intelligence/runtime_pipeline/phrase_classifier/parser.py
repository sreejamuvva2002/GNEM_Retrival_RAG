"""Parse and validate LLM JSON responses for phrase classification."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .models import (
    AmbiguousTerm,
    ClassifiedPhrase,
    PhraseCategory,
    PhraseClassificationResult,
    VALID_CATEGORIES,
)

logger = logging.getLogger(__name__)

# Fields that must NEVER appear in the response.
_FORBIDDEN_KEYS = frozenset({"rewritten_query", "filters", "sql", "rewrite"})


def parse_response(
    raw: str,
    original_phrases: list[str],
) -> PhraseClassificationResult:
    """Parse the LLM JSON response into a PhraseClassificationResult.

    Falls back to conservative classification on any parse failure.
    """
    if not raw or not raw.strip():
        logger.info("Empty LLM response — using fallback classification")
        return _fallback_classify(original_phrases, raw_response=raw)

    json_str = _extract_json(raw)
    if json_str is None:
        logger.warning("No JSON found in LLM response — using fallback")
        return _fallback_classify(original_phrases, raw_response=raw)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON in LLM response: %s — using fallback", exc)
        return _fallback_classify(original_phrases, raw_response=raw)

    if not isinstance(data, dict):
        logger.warning("LLM response is not a JSON object — using fallback")
        return _fallback_classify(original_phrases, raw_response=raw)

    # Reject forbidden fields.
    for key in _FORBIDDEN_KEYS:
        if key in data:
            logger.warning("LLM response contains forbidden field '%s' — removing it", key)
            data.pop(key)

    return _build_result(data, original_phrases, raw)


def _extract_json(text: str) -> str | None:
    """Extract JSON object from text, handling markdown fences."""
    # Try raw parse first.
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped

    # Try extracting from markdown code fence.
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if match:
        return match.group(1)

    # Try finding first { ... } block.
    start = stripped.find("{")
    if start == -1:
        return None
    end = stripped.rfind("}")
    if end <= start:
        return None
    return stripped[start : end + 1]


def _build_result(
    data: dict[str, Any],
    original_phrases: list[str],
    raw: str,
) -> PhraseClassificationResult:
    """Build result from parsed JSON, validating all fields."""
    classified: list[ClassifiedPhrase] = []
    ambiguous: list[AmbiguousTerm] = []
    semantic_intent_terms: list[str] = []
    context_terms: list[str] = []
    connector_terms: list[str] = []
    domain_signal_terms: list[str] = []
    irrelevant_terms: list[str] = []

    seen_phrases: set[str] = set()
    original_lower = {p.lower().strip() for p in original_phrases}

    for entry in data.get("classified_phrases", []):
        if not isinstance(entry, dict):
            continue
        phrase = str(entry.get("phrase", "")).strip()
        if not phrase:
            continue

        # Skip phrases the LLM hallucinated (not in original list).
        if phrase.lower() not in original_lower:
            logger.debug("Dropping hallucinated phrase: %r", phrase)
            continue

        if phrase.lower() in seen_phrases:
            continue
        seen_phrases.add(phrase.lower())

        category_str = str(entry.get("category", "")).strip()
        category = _validate_category(category_str)

        needs_clarification = bool(entry.get("needs_clarification", False))
        clarification_question = entry.get("clarification_question")
        reason = str(entry.get("reason", ""))

        # Enforce: needs_clarification only for ambiguous_concept.
        if category != PhraseCategory.ambiguous_concept:
            needs_clarification = False
            clarification_question = None

        if category == PhraseCategory.ambiguous_concept and not needs_clarification:
            needs_clarification = True

        if needs_clarification and not clarification_question:
            clarification_question = f'What do you mean by "{phrase}" in this query?'

        cp = ClassifiedPhrase(
            phrase=phrase,
            category=category,
            needs_clarification=needs_clarification,
            clarification_question=str(clarification_question) if clarification_question else None,
            reason=reason,
        )
        classified.append(cp)

        if needs_clarification:
            ambiguous.append(
                AmbiguousTerm(phrase=phrase, clarification_question=cp.clarification_question or "")
            )

        # Bucket into lists.
        _bucket_phrase(cp, semantic_intent_terms, context_terms, connector_terms,
                       domain_signal_terms, irrelevant_terms)

    # Handle phrases the LLM forgot to classify.
    for phrase in original_phrases:
        if phrase.lower().strip() not in seen_phrases:
            logger.debug("LLM missed phrase %r — classifying as semantic_intent", phrase)
            cp = ClassifiedPhrase(
                phrase=phrase,
                category=PhraseCategory.semantic_intent,
                needs_clarification=False,
                clarification_question=None,
                reason="Not classified by LLM — defaulting to semantic_intent",
            )
            classified.append(cp)
            semantic_intent_terms.append(phrase)

    # Extract top-level fields from LLM response.
    target_entity_override = data.get("target_entity_override")
    if target_entity_override is not None:
        target_entity_override = str(target_entity_override).strip() or None

    clarification_required = any(cp.needs_clarification for cp in classified)

    # Use LLM list fields only as cross-check — our bucketed lists are authoritative.
    return PhraseClassificationResult(
        classified_phrases=classified,
        clarification_required=clarification_required,
        ambiguous_terms=ambiguous,
        target_entity_override=target_entity_override,
        semantic_intent_terms=semantic_intent_terms,
        context_terms=context_terms,
        connector_terms=connector_terms,
        domain_signal_terms=domain_signal_terms,
        irrelevant_terms=irrelevant_terms,
        raw_response=raw,
        debug={"parsed_from_llm": True, "classified_count": len(classified)},
    )


def _validate_category(category_str: str) -> PhraseCategory:
    """Validate and return a PhraseCategory, defaulting to semantic_intent."""
    if category_str in VALID_CATEGORIES:
        return PhraseCategory(category_str)
    logger.warning("Invalid category %r — defaulting to semantic_intent", category_str)
    return PhraseCategory.semantic_intent


def _bucket_phrase(
    cp: ClassifiedPhrase,
    semantic_intent_terms: list[str],
    context_terms: list[str],
    connector_terms: list[str],
    domain_signal_terms: list[str],
    irrelevant_terms: list[str],
) -> None:
    """Add phrase to the appropriate bucket list."""
    if cp.category == PhraseCategory.semantic_intent:
        semantic_intent_terms.append(cp.phrase)
    elif cp.category == PhraseCategory.context_description:
        context_terms.append(cp.phrase)
    elif cp.category == PhraseCategory.intent_connector:
        connector_terms.append(cp.phrase)
    elif cp.category == PhraseCategory.domain_signal:
        domain_signal_terms.append(cp.phrase)
    elif cp.category == PhraseCategory.irrelevant_phrase:
        irrelevant_terms.append(cp.phrase)
    # target_entity and ambiguous_concept are not bucketed into separate lists here.


def _fallback_classify(
    original_phrases: list[str],
    raw_response: str | None = None,
) -> PhraseClassificationResult:
    """Conservative fallback: classify all phrases as semantic_intent."""
    classified = [
        ClassifiedPhrase(
            phrase=phrase,
            category=PhraseCategory.semantic_intent,
            needs_clarification=False,
            clarification_question=None,
            reason="Fallback classification — LLM response could not be parsed",
        )
        for phrase in original_phrases
    ]
    return PhraseClassificationResult(
        classified_phrases=classified,
        clarification_required=False,
        ambiguous_terms=[],
        target_entity_override=None,
        semantic_intent_terms=list(original_phrases),
        context_terms=[],
        connector_terms=[],
        domain_signal_terms=[],
        irrelevant_terms=[],
        raw_response=raw_response,
        debug={"parsed_from_llm": False, "fallback": True},
    )
