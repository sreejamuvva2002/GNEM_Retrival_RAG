"""Unit tests for deterministic query analysis."""
from __future__ import annotations

import pytest

from georgia_ev_intelligence.runtime_pipeline.query_analyzer import (
    InMemoryVocabularyRepository,
    QueryAnalyzer,
    VocabularyTerm,
)


def _term(
    normalized: str,
    *,
    canonical: str | None = None,
    source_column: str = "ev_supply_chain_role",
    term_type: str | None = None,
    term_id: int | None = None,
) -> VocabularyTerm:
    return VocabularyTerm(
        id=term_id,
        canonical_value=canonical or normalized.title(),
        normalized_value=normalized,
        source_column=source_column,
        term_type=term_type,
        aliases=[],
        row_ids=[],
    )


@pytest.fixture
def vocabulary_terms() -> list[VocabularyTerm]:
    return [
        _term("tier 1/2", canonical="Tier 1/2", source_column="category", term_type="supplier_tier", term_id=1),
        _term("tier 2/3", canonical="Tier 2/3", source_column="category", term_type="supplier_tier", term_id=2),
        _term("georgia", canonical="Georgia", source_column="updated_location", term_type="location", term_id=3),
        _term("battery recycling", canonical="Battery recycling", source_column="ev_supply_chain_role", term_type="ev_supply_chain_role", term_id=4),
        _term("battery", canonical="Battery", source_column="product_service", term_type="product_service", term_id=5),
        _term("battery cell", canonical="Battery cell", source_column="product_service", term_type="product_service", term_id=6),
    ]


@pytest.fixture
def analyzer(vocabulary_terms: list[VocabularyTerm]) -> QueryAnalyzer:
    return QueryAnalyzer(InMemoryVocabularyRepository(vocabulary_terms))


def _matched_values(result) -> list[str]:
    return [m.canonical_value for m in result.matched_vocabulary]


def test_tier_and_location_no_ambiguity(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("Show Tier 1/2 suppliers in Georgia")
    assert result.operation == "list"
    assert result.target_entity == "suppliers"
    assert "Tier 1/2" in _matched_values(result)
    assert "Georgia" in _matched_values(result)
    assert result.ambiguous_terms == []


def test_capacity_fragile_ambiguous(analyzer: QueryAnalyzer) -> None:
    query = (
        "Show capacity-fragile Tier 1/2 suppliers in Georgia "
        "involved in battery recycling"
    )
    result = analyzer.analyze(query)
    assert result.operation == "list"
    assert result.target_entity == "suppliers"
    matched = _matched_values(result)
    assert "Tier 1/2" in matched
    assert "Georgia" in matched
    assert "Battery recycling" in matched
    assert result.ambiguous_terms == ["capacity fragile"]
    assert "show" in result.ignored_tokens
    assert "in" in result.ignored_tokens
    assert "involved" in result.ignored_tokens


def test_longest_match_battery_recycling(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("List battery recycling companies")
    assert result.operation == "list"
    assert result.target_entity == "companies"
    matched = _matched_values(result)
    assert "Battery recycling" in matched
    assert "Battery" not in matched
    assert result.ambiguous_terms == []


def test_small_battery_suppliers(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("Find small battery suppliers")
    assert result.operation == "list"
    assert result.target_entity == "suppliers"
    assert "Battery" in _matched_values(result)
    assert "small" in result.ambiguous_terms


def test_two_tier_spans(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("Show Tier 1/2 and Tier 2/3 suppliers")
    matched = _matched_values(result)
    assert matched.count("Tier 1/2") == 1
    assert matched.count("Tier 2/3") == 1
    assert result.target_entity == "suppliers"
    assert result.ambiguous_terms == []


def test_normalized_query_preserves_tier_slash(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("Show Tier 1/2 suppliers in Georgia")
    assert "tier 1/2" in result.normalized_query
    assert "1 2" not in result.normalized_query.replace("tier 1/2", "")


def test_debug_payload(analyzer: QueryAnalyzer) -> None:
    result = analyzer.analyze("Show Tier 1/2 suppliers in Georgia")
    assert "tokens" in result.debug
    assert "max_phrase_length" in result.debug
    assert result.debug["match_count"] >= 2
