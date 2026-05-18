"""Unit tests for the phrase classifier response parser."""
from __future__ import annotations

import json

import pytest

from georgia_ev_intelligence.runtime_pipeline.phrase_classifier.models import (
    PhraseCategory,
    PhraseClassificationResult,
)
from georgia_ev_intelligence.runtime_pipeline.phrase_classifier.parser import (
    parse_response,
    _fallback_classify,
)


def _valid_json(
    phrases: list[dict] | None = None,
    clarification_required: bool = False,
    ambiguous_terms: list[dict] | None = None,
    target_entity_override: str | None = None,
) -> str:
    """Build a valid JSON response string."""
    data = {
        "classified_phrases": phrases or [],
        "clarification_required": clarification_required,
        "ambiguous_terms": ambiguous_terms or [],
        "target_entity_override": target_entity_override,
        "semantic_intent_terms": [],
        "context_terms": [],
        "connector_terms": [],
        "domain_signal_terms": [],
        "irrelevant_terms": [],
    }
    return json.dumps(data)


class TestValidJsonParsing:
    def test_single_semantic_intent_phrase(self) -> None:
        raw = _valid_json(
            phrases=[
                {
                    "phrase": "existing infrastructure",
                    "category": "semantic_intent",
                    "needs_clarification": False,
                    "clarification_question": None,
                    "reason": "Describes reasoning user wants.",
                }
            ]
        )
        result = parse_response(raw, ["existing infrastructure"])
        assert len(result.classified_phrases) == 1
        cp = result.classified_phrases[0]
        assert cp.phrase == "existing infrastructure"
        assert cp.category == PhraseCategory.semantic_intent
        assert cp.needs_clarification is False
        assert result.clarification_required is False

    def test_ambiguous_concept_sets_clarification(self) -> None:
        raw = _valid_json(
            phrases=[
                {
                    "phrase": "capacity fragile",
                    "category": "ambiguous_concept",
                    "needs_clarification": True,
                    "clarification_question": 'What do you mean by "capacity fragile"?',
                    "reason": "Multiple interpretations possible.",
                }
            ],
            clarification_required=True,
            ambiguous_terms=[
                {"phrase": "capacity fragile", "clarification_question": 'What do you mean by "capacity fragile"?'}
            ],
        )
        result = parse_response(raw, ["capacity fragile"])
        assert result.clarification_required is True
        assert len(result.ambiguous_terms) == 1
        assert result.ambiguous_terms[0].phrase == "capacity fragile"

    def test_multiple_categories_bucketed(self) -> None:
        raw = _valid_json(
            phrases=[
                {"phrase": "seeking", "category": "intent_connector", "needs_clarification": False, "reason": "connector"},
                {"phrase": "chemical manufacturing", "category": "domain_signal", "needs_clarification": False, "reason": "domain"},
                {"phrase": "existing", "category": "context_description", "needs_clarification": False, "reason": "context"},
            ]
        )
        result = parse_response(raw, ["seeking", "chemical manufacturing", "existing"])
        assert "seeking" in result.connector_terms
        assert "chemical manufacturing" in result.domain_signal_terms
        assert "existing" in result.context_terms
        assert result.clarification_required is False

    def test_target_entity_override_extracted(self) -> None:
        raw = _valid_json(
            phrases=[
                {"phrase": "areas", "category": "target_entity", "needs_clarification": False, "reason": "target entity"},
            ],
            target_entity_override="areas",
        )
        result = parse_response(raw, ["areas"])
        assert result.target_entity_override == "areas"

    def test_irrelevant_phrase_bucketed(self) -> None:
        raw = _valid_json(
            phrases=[
                {"phrase": "please", "category": "irrelevant_phrase", "needs_clarification": False, "reason": "noise"},
            ]
        )
        result = parse_response(raw, ["please"])
        assert "please" in result.irrelevant_terms


class TestNeedsClarificationEnforcement:
    def test_needs_clarification_only_for_ambiguous_concept(self) -> None:
        """If LLM sets needs_clarification=True for non-ambiguous, parser corrects it."""
        raw = _valid_json(
            phrases=[
                {
                    "phrase": "chemical manufacturing",
                    "category": "domain_signal",
                    "needs_clarification": True,
                    "clarification_question": "Some question",
                    "reason": "domain",
                }
            ]
        )
        result = parse_response(raw, ["chemical manufacturing"])
        cp = result.classified_phrases[0]
        assert cp.needs_clarification is False
        assert cp.clarification_question is None

    def test_ambiguous_concept_forced_needs_clarification(self) -> None:
        """If LLM classifies as ambiguous but forgets needs_clarification, parser forces it."""
        raw = _valid_json(
            phrases=[
                {
                    "phrase": "fragile",
                    "category": "ambiguous_concept",
                    "needs_clarification": False,
                    "clarification_question": None,
                    "reason": "ambiguous",
                }
            ]
        )
        result = parse_response(raw, ["fragile"])
        cp = result.classified_phrases[0]
        assert cp.needs_clarification is True
        assert cp.clarification_question is not None  # auto-generated


class TestRewrittenQueryRejection:
    def test_rewritten_query_field_removed(self) -> None:
        """Response containing rewritten_query is sanitized, not rejected."""
        data = {
            "rewritten_query": "This should be removed",
            "classified_phrases": [
                {"phrase": "existing", "category": "context_description", "needs_clarification": False, "reason": "ok"},
            ],
            "clarification_required": False,
            "ambiguous_terms": [],
        }
        raw = json.dumps(data)
        result = parse_response(raw, ["existing"])
        assert len(result.classified_phrases) == 1
        assert result.classified_phrases[0].phrase == "existing"

    def test_filters_field_removed(self) -> None:
        data = {
            "filters": {"column": "value"},
            "classified_phrases": [],
            "clarification_required": False,
            "ambiguous_terms": [],
        }
        raw = json.dumps(data)
        result = parse_response(raw, [])
        assert result.clarification_required is False


class TestInvalidCategoryFallback:
    def test_invalid_category_defaults_to_semantic_intent(self) -> None:
        raw = _valid_json(
            phrases=[
                {"phrase": "test phrase", "category": "made_up_category", "needs_clarification": False, "reason": "bad"},
            ]
        )
        result = parse_response(raw, ["test phrase"])
        assert result.classified_phrases[0].category == PhraseCategory.semantic_intent


class TestMalformedJson:
    def test_empty_string_uses_fallback(self) -> None:
        result = parse_response("", ["term1", "term2"])
        assert result.debug.get("fallback") is True
        assert len(result.classified_phrases) == 2
        assert all(cp.category == PhraseCategory.semantic_intent for cp in result.classified_phrases)

    def test_garbage_text_uses_fallback(self) -> None:
        result = parse_response("this is not json at all!", ["term1"])
        assert result.debug.get("fallback") is True
        assert len(result.classified_phrases) == 1

    def test_partial_json_uses_fallback(self) -> None:
        result = parse_response('{"classified_phrases": [{"phrase": "abc"', ["abc"])
        assert result.debug.get("fallback") is True

    def test_json_in_markdown_fence(self) -> None:
        inner = _valid_json(
            phrases=[
                {"phrase": "test", "category": "domain_signal", "needs_clarification": False, "reason": "ok"},
            ]
        )
        raw = f"```json\n{inner}\n```"
        result = parse_response(raw, ["test"])
        assert result.classified_phrases[0].category == PhraseCategory.domain_signal


class TestMissingAndExtraPhrases:
    def test_missing_phrase_gets_default_classification(self) -> None:
        """Phrases the LLM forgot to classify get semantic_intent."""
        raw = _valid_json(
            phrases=[
                {"phrase": "term1", "category": "domain_signal", "needs_clarification": False, "reason": "ok"},
            ]
        )
        result = parse_response(raw, ["term1", "term2"])
        assert len(result.classified_phrases) == 2
        term2_cp = [cp for cp in result.classified_phrases if cp.phrase == "term2"][0]
        assert term2_cp.category == PhraseCategory.semantic_intent

    def test_extra_hallucinated_phrase_dropped(self) -> None:
        """Phrases the LLM hallucinated are dropped."""
        raw = _valid_json(
            phrases=[
                {"phrase": "real term", "category": "domain_signal", "needs_clarification": False, "reason": "ok"},
                {"phrase": "hallucinated", "category": "domain_signal", "needs_clarification": False, "reason": "fake"},
            ]
        )
        result = parse_response(raw, ["real term"])
        assert len(result.classified_phrases) == 1
        assert result.classified_phrases[0].phrase == "real term"


class TestEmptyInput:
    def test_no_phrases_empty_result(self) -> None:
        raw = _valid_json()
        result = parse_response(raw, [])
        assert result.classified_phrases == []
        assert result.clarification_required is False


class TestFallbackClassify:
    def test_fallback_classifies_all_as_semantic_intent(self) -> None:
        result = _fallback_classify(["term1", "term2"])
        assert len(result.classified_phrases) == 2
        assert all(cp.category == PhraseCategory.semantic_intent for cp in result.classified_phrases)
        assert result.clarification_required is False
        assert result.semantic_intent_terms == ["term1", "term2"]
