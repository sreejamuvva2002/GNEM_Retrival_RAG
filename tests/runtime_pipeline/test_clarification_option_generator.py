"""Unit tests for ClarificationOptionGenerator module."""
from __future__ import annotations

import pytest

from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator import (
    ClarificationAnswer,
    ClarificationOptionGenerator,
    ClarificationResolver,
    ClarificationSubmission,
    InMemoryClarificationStore,
    InMemoryConceptRegistry,
    JsonConceptRegistry,
)
from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator.concept_registry import (
    default_concepts_path,
)
from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator.exceptions import (
    ClarificationAlreadyResolvedError,
    InvalidClarificationAnswerError,
    MissingCustomClarificationTextError,
)
from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator.models import (
    ClarificationOption,
    ConceptDefinition,
)
from georgia_ev_intelligence.runtime_pipeline.query_analyzer.models import (
    QueryAnalysisResult,
    VocabularyMatch,
)


def _analysis(
    *,
    ambiguous: list[str] | None = None,
    matched: list[VocabularyMatch] | None = None,
    query: str = "test query",
) -> QueryAnalysisResult:
    return QueryAnalysisResult(
        original_query=query,
        normalized_query=query.lower(),
        operation="list",
        target_entity="suppliers",
        matched_vocabulary=matched or [],
        ambiguous_terms=ambiguous or [],
    )


def _capacity_concept() -> ConceptDefinition:
    registry = JsonConceptRegistry(default_concepts_path())
    concept = registry.find_concept("capacity fragile")
    assert concept is not None
    return concept


@pytest.fixture
def concept_registry() -> InMemoryConceptRegistry:
    return InMemoryConceptRegistry([_capacity_concept()])


@pytest.fixture
def store() -> InMemoryClarificationStore:
    return InMemoryClarificationStore()


@pytest.fixture
def generator(
    concept_registry: InMemoryConceptRegistry,
    store: InMemoryClarificationStore,
) -> ClarificationOptionGenerator:
    return ClarificationOptionGenerator(concept_registry, store=store)


def test_known_concept_generates_full_options(generator: ClarificationOptionGenerator) -> None:
    analysis = _analysis(ambiguous=["capacity fragile"])
    request = generator.generate_request("Show capacity-fragile suppliers", analysis)
    assert request is not None
    assert request.clarification_required is True
    assert len(request.ambiguous_terms) == 1
    term = request.ambiguous_terms[0]
    assert term.matched_concept_id == "capacity_fragility"
    option_ids = {o.id for o in term.options}
    assert option_ids >= {
        "low_employment",
        "limited_oems",
        "critical_small_scale",
        "combination",
        "custom",
        "ignore",
    }


def test_unknown_term_generic_options() -> None:
    registry = InMemoryConceptRegistry([_capacity_concept()])
    gen = ClarificationOptionGenerator(registry)
    request = gen.generate_request(
        "Show strategically weak suppliers in Georgia",
        _analysis(ambiguous=["strategically weak"]),
    )
    assert request is not None
    term = request.ambiguous_terms[0]
    assert term.matched_concept_id is None
    assert {o.id for o in term.options} == {"custom", "ignore"}


def test_no_ambiguous_returns_none(generator: ClarificationOptionGenerator) -> None:
    assert generator.generate_request("q", _analysis(ambiguous=[])) is None


def test_resolve_combination_option(
    generator: ClarificationOptionGenerator,
    store: InMemoryClarificationStore,
) -> None:
    analysis = _analysis(ambiguous=["capacity fragile"])
    request = generator.generate_request("Show capacity-fragile suppliers", analysis)
    assert request is not None
    resolver = ClarificationResolver(store)
    ctx = resolver.resolve(
        ClarificationSubmission(
            clarification_id=request.clarification_id,
            answers=[
                ClarificationAnswer(
                    term="capacity fragile",
                    selected_option_id="combination",
                )
            ],
        )
    )
    rc = ctx.resolved_clarifications[0]
    assert rc.selected_option_id == "combination"
    assert rc.meaning is not None
    assert "low employment" in rc.meaning.lower()
    assert "limited oem" in rc.meaning.lower()
    assert "critical" in rc.meaning.lower() or "small" in rc.meaning.lower()


def test_resolve_custom_option(store: InMemoryClarificationStore) -> None:
    registry = InMemoryConceptRegistry([_capacity_concept()])
    gen = ClarificationOptionGenerator(registry, store=store)
    request = gen.generate_request("q", _analysis(ambiguous=["capacity fragile"]))
    assert request is not None
    resolver = ClarificationResolver(store)
    ctx = resolver.resolve(
        ClarificationSubmission(
            clarification_id=request.clarification_id,
            answers=[
                ClarificationAnswer(
                    term="capacity fragile",
                    selected_option_id="custom",
                    custom_text="low employment and one listed OEM",
                )
            ],
        )
    )
    assert ctx.resolved_clarifications[0].meaning == "low employment and one listed OEM"


def test_resolve_ignore_option(
    generator: ClarificationOptionGenerator,
    store: InMemoryClarificationStore,
) -> None:
    request = generator.generate_request("q", _analysis(ambiguous=["capacity fragile"]))
    assert request is not None
    resolver = ClarificationResolver(store)
    ctx = resolver.resolve(
        ClarificationSubmission(
            clarification_id=request.clarification_id,
            answers=[
                ClarificationAnswer(
                    term="capacity fragile",
                    selected_option_id="ignore",
                )
            ],
        )
    )
    rc = ctx.resolved_clarifications[0]
    assert rc.ignored is True
    assert "ignore" in ctx.final_generation_notes[0].lower()


def test_invalid_option_id(
    generator: ClarificationOptionGenerator,
    store: InMemoryClarificationStore,
) -> None:
    request = generator.generate_request("q", _analysis(ambiguous=["capacity fragile"]))
    assert request is not None
    resolver = ClarificationResolver(store)
    with pytest.raises(InvalidClarificationAnswerError):
        resolver.resolve(
            ClarificationSubmission(
                clarification_id=request.clarification_id,
                answers=[
                    ClarificationAnswer(
                        term="capacity fragile",
                        selected_option_id="not_a_real_option",
                    )
                ],
            )
        )


def test_missing_custom_text(
    generator: ClarificationOptionGenerator,
    store: InMemoryClarificationStore,
) -> None:
    request = generator.generate_request("q", _analysis(ambiguous=["capacity fragile"]))
    assert request is not None
    resolver = ClarificationResolver(store)
    with pytest.raises(MissingCustomClarificationTextError):
        resolver.resolve(
            ClarificationSubmission(
                clarification_id=request.clarification_id,
                answers=[
                    ClarificationAnswer(
                        term="capacity fragile",
                        selected_option_id="custom",
                    )
                ],
            )
        )


class FakeBatteryRecyclingAnalyzer:
    def analyze(self, query: str) -> QueryAnalysisResult:
        matched = []
        if "battery recycling" in query.lower():
            matched.append(
                VocabularyMatch(
                    matched_text="battery recycling",
                    canonical_value="Battery recycling",
                    source_column="EV Supply Chain Role",
                    term_type="ev_supply_chain_role",
                    match_type="canonical",
                    start_token=0,
                    end_token=2,
                )
            )
        return QueryAnalysisResult(
            original_query=query,
            normalized_query=query.lower(),
            operation="list",
            target_entity="suppliers",
            matched_vocabulary=matched,
            ambiguous_terms=["low employment"] if "low employment" in query.lower() else [],
        )


def test_clarification_reanalysis_merges_vocabulary(
    store: InMemoryClarificationStore,
) -> None:
    registry = InMemoryConceptRegistry([_capacity_concept()])
    gen = ClarificationOptionGenerator(registry, store=store)
    original = _analysis(
        ambiguous=["fragile"],
        matched=[
            VocabularyMatch(
                matched_text="georgia",
                canonical_value="Georgia",
                source_column="Location",
                term_type="location",
                match_type="canonical",
                start_token=0,
                end_token=1,
            )
        ],
        query="Show fragile suppliers in Georgia",
    )
    request = gen.generate_request(original.original_query, original)
    assert request is not None

    resolver = ClarificationResolver(store, query_analyzer=FakeBatteryRecyclingAnalyzer())
    ctx = resolver.resolve(
        ClarificationSubmission(
            clarification_id=request.clarification_id,
            answers=[
                ClarificationAnswer(
                    term="fragile",
                    selected_option_id="custom",
                    custom_text="I mean battery recycling suppliers with low employment.",
                )
            ],
        )
    )
    canonical_values = [
        m.canonical_value for m in ctx.merged_matched_vocabulary
    ]
    assert "Georgia" in canonical_values
    assert "Battery recycling" in canonical_values
    assert "low employment" in ctx.remaining_ambiguous_terms


def test_deduplicate_georgia_in_merge(store: InMemoryClarificationStore) -> None:
    georgia = VocabularyMatch(
        matched_text="georgia",
        canonical_value="Georgia",
        source_column="Location",
        term_type="location",
        match_type="canonical",
        start_token=0,
        end_token=1,
    )
    original = _analysis(matched=[georgia], ambiguous=["fragile"])
    registry = InMemoryConceptRegistry([_capacity_concept()])
    gen = ClarificationOptionGenerator(registry, store=store)
    request = gen.generate_request("Show fragile suppliers in Georgia", original)
    assert request is not None

    class SameGeorgiaAnalyzer:
        def analyze(self, query: str) -> QueryAnalysisResult:
            return _analysis(matched=[georgia])

    resolver = ClarificationResolver(store, query_analyzer=SameGeorgiaAnalyzer())
    ctx = resolver.resolve(
        ClarificationSubmission(
            clarification_id=request.clarification_id,
            answers=[
                ClarificationAnswer(
                    term="fragile",
                    selected_option_id="ignore",
                )
            ],
        )
    )
    georgia_count = sum(
        1
        for m in ctx.merged_matched_vocabulary
        if getattr(m, "canonical_value", None) == "Georgia"
    )
    assert georgia_count == 1


def test_double_resolve_raises(
    generator: ClarificationOptionGenerator,
    store: InMemoryClarificationStore,
) -> None:
    request = generator.generate_request("q", _analysis(ambiguous=["capacity fragile"]))
    assert request is not None
    resolver = ClarificationResolver(store)
    submission = ClarificationSubmission(
        clarification_id=request.clarification_id,
        answers=[
            ClarificationAnswer(term="capacity fragile", selected_option_id="ignore")
        ],
    )
    resolver.resolve(submission)
    with pytest.raises(ClarificationAlreadyResolvedError):
        resolver.resolve(submission)
