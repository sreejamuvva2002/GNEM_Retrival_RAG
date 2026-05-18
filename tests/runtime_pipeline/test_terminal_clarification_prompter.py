"""Tests for terminal clarification prompter and workflow."""
from __future__ import annotations

from typing import Any

import pytest

from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator import (
    ClarificationOptionGenerator,
    ClarificationResolver,
    InMemoryClarificationStore,
    InMemoryConceptRegistry,
    JsonConceptRegistry,
    TerminalClarificationPrompter,
    TerminalClarificationWorkflow,
)
from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator.concept_registry import (
    default_concepts_path,
)
from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator.constants import (
    DEFAULT_UNKNOWN_TERM_OPTIONS,
)
from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator.exceptions import (
    ClarificationCancelledError,
)
from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator.models import (
    AmbiguousTermClarification,
    ClarificationOption,
    ClarificationRequest,
    ConceptDefinition,
)
from georgia_ev_intelligence.runtime_pipeline.query_analyzer.models import (
    QueryAnalysisResult,
    VocabularyMatch,
)


def _capacity_request(clarification_id: str = "test-id") -> ClarificationRequest:
    concept = JsonConceptRegistry(default_concepts_path()).find_concept("capacity fragile")
    assert concept is not None
    return ClarificationRequest(
        clarification_required=True,
        clarification_id=clarification_id,
        original_query="Show capacity-fragile Tier 1/2 suppliers in Georgia involved in battery recycling",
        ambiguous_terms=[
            AmbiguousTermClarification(
                term="capacity fragile",
                normalized_term="capacity fragile",
                matched_concept_id=concept.concept_id,
                question='How should I interpret "capacity fragile"?',
                options=list(concept.clarification_options),
            )
        ],
    )


def _fake_io(inputs: list[str]) -> tuple[list[str], list[str]]:
    outputs: list[str] = []
    index = {"i": 0}

    def input_func(_prompt: str) -> str:
        if index["i"] >= len(inputs):
            raise AssertionError(f"Unexpected input prompt; exhausted inputs: {inputs}")
        value = inputs[index["i"]]
        index["i"] += 1
        return value

    def output_func(text: str) -> None:
        outputs.append(text)

    return input_func, output_func, outputs


def test_select_combination_option() -> None:
    input_func, output_func, outputs = _fake_io(["4"])
    prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
    submission = prompter.prompt(_capacity_request())
    assert submission.answers[0].selected_option_id == "combination"
    assert submission.answers[0].custom_text is None
    assert "Clarification Required" in "\n".join(outputs)
    assert "Enter option number" in "\n".join(outputs)


def test_custom_option_with_text() -> None:
    input_func, output_func, _ = _fake_io(["5", "low employment and one listed OEM"])
    prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
    submission = prompter.prompt(_capacity_request())
    assert submission.answers[0].selected_option_id == "custom"
    assert submission.answers[0].custom_text == "low employment and one listed OEM"


def test_ignore_option() -> None:
    input_func, output_func, _ = _fake_io(["6"])
    prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
    submission = prompter.prompt(_capacity_request())
    assert submission.answers[0].selected_option_id == "ignore"


def test_invalid_input_then_valid() -> None:
    input_func, output_func, outputs = _fake_io(["abc", "99", "4"])
    prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
    submission = prompter.prompt(_capacity_request())
    assert submission.answers[0].selected_option_id == "combination"
    joined = "\n".join(outputs)
    assert "Invalid option" in joined


def test_custom_empty_then_valid() -> None:
    input_func, output_func, outputs = _fake_io(["5", "", "battery recycling suppliers"])
    prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
    submission = prompter.prompt(_capacity_request())
    assert submission.answers[0].custom_text == "battery recycling suppliers"
    assert "cannot be empty" in "\n".join(outputs).lower()


def test_user_cancel() -> None:
    input_func, output_func, _ = _fake_io(["q"])
    prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
    with pytest.raises(ClarificationCancelledError):
        prompter.prompt(_capacity_request())


def test_multiple_ambiguous_terms() -> None:
    concept = JsonConceptRegistry(default_concepts_path()).find_concept("capacity fragile")
    assert concept is not None
    request = ClarificationRequest(
        clarification_required=True,
        clarification_id="multi-id",
        original_query="q",
        ambiguous_terms=[
            AmbiguousTermClarification(
                term="capacity fragile",
                normalized_term="capacity fragile",
                matched_concept_id="capacity_fragility",
                question="Q1",
                options=list(concept.clarification_options),
            ),
            AmbiguousTermClarification(
                term="strategic risk",
                normalized_term="strategic risk",
                matched_concept_id=None,
                question="Q2",
                options=list(DEFAULT_UNKNOWN_TERM_OPTIONS),
            ),
        ],
    )
    input_func, output_func, _ = _fake_io(["4", "2"])
    prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
    submission = prompter.prompt(request)
    assert len(submission.answers) == 2
    assert submission.answers[0].selected_option_id == "combination"
    assert submission.answers[1].selected_option_id == "ignore"


class TrackingAnalyzer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def analyze(self, query: str) -> QueryAnalysisResult:
        self.calls.append(query)
        if "battery recycling" in query.lower():
            return QueryAnalysisResult(
                original_query=query,
                normalized_query=query.lower(),
                operation="list",
                target_entity="suppliers",
                matched_vocabulary=[
                    VocabularyMatch(
                        matched_text="battery recycling",
                        canonical_value="Battery recycling",
                        source_column="EV Supply Chain Role",
                        term_type="ev_supply_chain_role",
                        match_type="canonical",
                        start_token=0,
                        end_token=2,
                    )
                ],
                ambiguous_terms=["low employment"],
            )
        return QueryAnalysisResult(
            original_query=query,
            normalized_query=query.lower(),
            operation="list",
            target_entity="suppliers",
            matched_vocabulary=[
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
            ambiguous_terms=["fragile"],
        )


def test_resolver_reanalyzes_custom_clarification() -> None:
    store = InMemoryClarificationStore()
    registry = InMemoryConceptRegistry(
        [JsonConceptRegistry(default_concepts_path()).find_concept("capacity fragile")]  # type: ignore[list-item]
    )
    gen = ClarificationOptionGenerator(registry, store=store)
    analysis = QueryAnalysisResult(
        original_query="Show fragile suppliers in Georgia",
        normalized_query="show fragile suppliers in georgia",
        operation="list",
        target_entity="suppliers",
        matched_vocabulary=[
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
        ambiguous_terms=["fragile"],
    )
    request = gen.generate_request(analysis.original_query, analysis)
    assert request is not None

    unknown_request = ClarificationRequest(
        clarification_required=True,
        clarification_id=request.clarification_id,
        original_query=analysis.original_query,
        ambiguous_terms=[
            AmbiguousTermClarification(
                term="fragile",
                normalized_term="fragile",
                matched_concept_id=None,
                question="Q",
                options=list(DEFAULT_UNKNOWN_TERM_OPTIONS),
            )
        ],
    )
    session = store.get_session(request.clarification_id)
    session.request = unknown_request
    store.update_session(session)

    analyzer = TrackingAnalyzer()
    resolver = ClarificationResolver(store, query_analyzer=analyzer)
    input_func, output_func, _ = _fake_io(["1", "battery recycling suppliers with low employment"])
    prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
    submission = prompter.prompt(unknown_request)
    ctx = resolver.resolve(submission)

    canonical = [m.canonical_value for m in ctx.merged_matched_vocabulary]
    assert "Georgia" in canonical
    assert "Battery recycling" in canonical
    assert len(analyzer.calls) == 1
    assert "battery recycling" in analyzer.calls[0].lower()


def test_resolver_skips_analyzer_for_ignore() -> None:
    store = InMemoryClarificationStore()
    registry = JsonConceptRegistry(default_concepts_path())
    gen = ClarificationOptionGenerator(registry, store=store)
    request = gen.generate_request(
        "q",
        QueryAnalysisResult(
            original_query="q",
            normalized_query="q",
            operation="list",
            target_entity="suppliers",
            ambiguous_terms=["capacity fragile"],
        ),
    )
    assert request is not None
    analyzer = TrackingAnalyzer()
    resolver = ClarificationResolver(store, query_analyzer=analyzer)
    input_func, output_func, _ = _fake_io(["6"])
    prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
    submission = prompter.prompt(request)
    ctx = resolver.resolve(submission)
    assert ctx.resolved_clarifications[0].ignored is True
    assert analyzer.calls == []


def test_workflow_end_to_end() -> None:
    store = InMemoryClarificationStore()
    registry = JsonConceptRegistry(default_concepts_path())
    analyzer = TrackingAnalyzer()
    gen = ClarificationOptionGenerator(registry, store=store)
    input_func, output_func, outputs = _fake_io(
        ["1", "battery recycling suppliers with low employment"]
    )
    prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
    resolver = ClarificationResolver(store, query_analyzer=analyzer)
    workflow = TerminalClarificationWorkflow(analyzer, gen, prompter, resolver)

    ctx = workflow.handle_query("Show fragile suppliers in Georgia")
    assert any("clarified" in note.lower() for note in ctx.final_generation_notes)
    assert "Clarification Required" in "\n".join(outputs)
    assert "Georgia" in [m.canonical_value for m in ctx.merged_matched_vocabulary]
