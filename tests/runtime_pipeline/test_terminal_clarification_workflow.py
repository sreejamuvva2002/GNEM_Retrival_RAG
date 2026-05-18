"""Tests for the open-ended terminal clarification workflow."""
from __future__ import annotations

from typing import Any

import pytest

from georgia_ev_intelligence.runtime_pipeline.query_analyzer.models import (
    QueryAnalysisResult,
    VocabularyMatch,
)
from georgia_ev_intelligence.runtime_pipeline.phrase_classifier.models import (
    AmbiguousTerm,
    ClassifiedPhrase,
    PhraseCategory,
    PhraseClassificationResult,
)
from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator import (
    AnalysisMerger,
    ClarificationResolver,
    InMemoryClarificationStore,
    TerminalClarificationPrompter,
    TerminalClarificationWorkflow,
)
from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator.exceptions import (
    ClarificationAlreadyResolvedError,
    ClarificationCancelledError,
)
from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator.models import (
    ClarificationRequest,
)


# --- Fakes ---


class FakeAnalyzer:
    """Fake QueryAnalyzer that returns preconfigured analysis."""

    def __init__(
        self,
        matched: list[VocabularyMatch] | None = None,
        ambiguous: list[str] | None = None,
        operation: str | None = "list",
        target_entity: str | None = "suppliers",
    ) -> None:
        self._matched = matched or []
        self._ambiguous = ambiguous or []
        self._operation = operation
        self._target_entity = target_entity
        self.calls: list[str] = []

    def analyze(self, query: str) -> QueryAnalysisResult:
        self.calls.append(query)
        matched = list(self._matched)
        ambiguous = list(self._ambiguous)
        # If query mentions "battery recycling", produce a match for it.
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
            operation=self._operation,
            target_entity=self._target_entity,
            matched_vocabulary=matched,
            ambiguous_terms=ambiguous,
        )


class FakeClassifier:
    """Fake RemainingPhraseClassifier with configurable behavior."""

    def __init__(
        self,
        clarification_required: bool = False,
        ambiguous_terms: list[AmbiguousTerm] | None = None,
        semantic_intent_terms: list[str] | None = None,
    ) -> None:
        self._clarification_required = clarification_required
        self._ambiguous = ambiguous_terms or []
        self._semantic = semantic_intent_terms or []
        self.calls: list[tuple[str, Any]] = []

    def classify(self, original_query: str, analysis: Any) -> PhraseClassificationResult:
        self.calls.append((original_query, analysis))
        classified: list[ClassifiedPhrase] = []
        for at in self._ambiguous:
            classified.append(ClassifiedPhrase(
                phrase=at.phrase,
                category=PhraseCategory.ambiguous_concept,
                needs_clarification=True,
                clarification_question=at.clarification_question,
                reason="test",
            ))
        for s in self._semantic:
            classified.append(ClassifiedPhrase(
                phrase=s,
                category=PhraseCategory.semantic_intent,
                needs_clarification=False,
                clarification_question=None,
                reason="test",
            ))
        return PhraseClassificationResult(
            classified_phrases=classified,
            clarification_required=self._clarification_required,
            ambiguous_terms=list(self._ambiguous),
            semantic_intent_terms=list(self._semantic),
        )


def _fake_io(inputs: list[str]):
    """Create fake input/output functions for terminal prompting."""
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


# --- Tests ---


class TestNoClarificationNeeded:
    def test_returns_resolved_context_without_prompting(self) -> None:
        analyzer = FakeAnalyzer(
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
            ambiguous=["existing infrastructure"],
        )
        classifier = FakeClassifier(
            clarification_required=False,
            semantic_intent_terms=["existing infrastructure"],
        )
        store = InMemoryClarificationStore()
        input_func, output_func, outputs = _fake_io([])
        prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
        resolver = ClarificationResolver(store=store, query_analyzer=analyzer, phrase_classifier=classifier)
        workflow = TerminalClarificationWorkflow(
            analyzer=analyzer,
            phrase_classifier=classifier,
            store=store,
            prompter=prompter,
            resolver=resolver,
        )

        ctx = workflow.analyze_and_maybe_clarify("Show companies in Georgia with existing infrastructure")
        assert ctx.resolved_clarifications == []
        assert "existing infrastructure" in ctx.semantic_intent_terms
        assert ctx.operation == "list"
        assert "Georgia" in [m.canonical_value for m in ctx.merged_matched_vocabulary]
        # No terminal output for prompting.
        assert "Clarification Required" not in "\n".join(outputs)


class TestClarificationNeeded:
    def test_displays_terminal_prompt(self) -> None:
        analyzer = FakeAnalyzer(ambiguous=["capacity fragile"])
        classifier = FakeClassifier(
            clarification_required=True,
            ambiguous_terms=[
                AmbiguousTerm(
                    phrase="capacity fragile",
                    clarification_question='What do you mean by "capacity fragile" in this query?',
                )
            ],
        )
        store = InMemoryClarificationStore()
        input_func, output_func, outputs = _fake_io(["low employment and limited OEM customers"])
        prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
        resolver = ClarificationResolver(store=store, query_analyzer=analyzer, phrase_classifier=classifier)
        workflow = TerminalClarificationWorkflow(
            analyzer=analyzer,
            phrase_classifier=classifier,
            store=store,
            prompter=prompter,
            resolver=resolver,
        )

        ctx = workflow.analyze_and_maybe_clarify("Show capacity-fragile suppliers")
        joined = "\n".join(outputs)
        assert "Clarification Required" in joined
        assert '"capacity fragile"' in joined
        assert len(ctx.resolved_clarifications) == 1
        assert ctx.resolved_clarifications[0].meaning == "low employment and limited OEM customers"

    def test_user_clarification_re_analyzed(self) -> None:
        """Resolver re-runs QueryAnalyzer on clarification text."""
        georgia_match = VocabularyMatch(
            matched_text="georgia",
            canonical_value="Georgia",
            source_column="Location",
            term_type="location",
            match_type="canonical",
            start_token=0,
            end_token=1,
        )
        analyzer = FakeAnalyzer(matched=[georgia_match], ambiguous=["fragile"])
        classifier = FakeClassifier(
            clarification_required=True,
            ambiguous_terms=[
                AmbiguousTerm(
                    phrase="fragile",
                    clarification_question='What do you mean by "fragile"?',
                )
            ],
        )
        store = InMemoryClarificationStore()
        input_func, output_func, _ = _fake_io(["battery recycling suppliers with low employment"])
        prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
        resolver = ClarificationResolver(
            store=store, query_analyzer=analyzer, phrase_classifier=classifier
        )
        workflow = TerminalClarificationWorkflow(
            analyzer=analyzer,
            phrase_classifier=classifier,
            store=store,
            prompter=prompter,
            resolver=resolver,
        )

        ctx = workflow.analyze_and_maybe_clarify("Show fragile suppliers in Georgia")

        canonical_values = [m.canonical_value for m in ctx.merged_matched_vocabulary]
        assert "Georgia" in canonical_values
        assert "Battery recycling" in canonical_values
        # Analyzer should have been called twice: once for original, once for clarification.
        assert len(analyzer.calls) == 2
        assert "battery recycling" in analyzer.calls[1].lower()


class TestNoSecondClarificationLoop:
    def test_no_second_clarification_triggered(self) -> None:
        """Even if clarification analysis has ambiguous terms, no second loop."""
        analyzer = FakeAnalyzer(ambiguous=["fragile"])
        classifier = FakeClassifier(
            clarification_required=True,
            ambiguous_terms=[
                AmbiguousTerm(phrase="fragile", clarification_question="What?")
            ],
        )
        store = InMemoryClarificationStore()
        input_func, output_func, outputs = _fake_io(["means low employment"])
        prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
        resolver = ClarificationResolver(
            store=store, query_analyzer=analyzer, phrase_classifier=classifier
        )
        workflow = TerminalClarificationWorkflow(
            analyzer=analyzer,
            phrase_classifier=classifier,
            store=store,
            prompter=prompter,
            resolver=resolver,
        )

        ctx = workflow.analyze_and_maybe_clarify("Show fragile suppliers")
        # Only one "Clarification Required" header should appear.
        header_count = sum(1 for o in outputs if o == "Clarification Required")
        assert header_count == 1


class TestCancelInput:
    def test_cancel_raises_error(self) -> None:
        analyzer = FakeAnalyzer(ambiguous=["fragile"])
        classifier = FakeClassifier(
            clarification_required=True,
            ambiguous_terms=[
                AmbiguousTerm(phrase="fragile", clarification_question="What?")
            ],
        )
        store = InMemoryClarificationStore()
        input_func, output_func, _ = _fake_io(["q"])
        prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
        resolver = ClarificationResolver(store=store, query_analyzer=analyzer)
        workflow = TerminalClarificationWorkflow(
            analyzer=analyzer,
            phrase_classifier=classifier,
            store=store,
            prompter=prompter,
            resolver=resolver,
        )

        with pytest.raises(ClarificationCancelledError):
            workflow.analyze_and_maybe_clarify("Show fragile suppliers")


class TestEmptyClarificationReprompts:
    def test_empty_input_reprompts(self) -> None:
        analyzer = FakeAnalyzer(ambiguous=["fragile"])
        classifier = FakeClassifier(
            clarification_required=True,
            ambiguous_terms=[
                AmbiguousTerm(phrase="fragile", clarification_question="What?")
            ],
        )
        store = InMemoryClarificationStore()
        # First empty, then valid.
        input_func, output_func, outputs = _fake_io(["", "low employment"])
        prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
        resolver = ClarificationResolver(store=store, query_analyzer=analyzer, phrase_classifier=classifier)
        workflow = TerminalClarificationWorkflow(
            analyzer=analyzer,
            phrase_classifier=classifier,
            store=store,
            prompter=prompter,
            resolver=resolver,
        )

        ctx = workflow.analyze_and_maybe_clarify("Show fragile suppliers")
        joined = "\n".join(outputs)
        assert "cannot be empty" in joined.lower()
        assert ctx.resolved_clarifications[0].meaning == "low employment"


class TestMergerIncludesVocabularyFromClarification:
    def test_merged_vocabulary_includes_clarification_matches(self) -> None:
        georgia_match = VocabularyMatch(
            matched_text="georgia",
            canonical_value="Georgia",
            source_column="Location",
            term_type="location",
            match_type="canonical",
            start_token=0,
            end_token=1,
        )
        analyzer = FakeAnalyzer(matched=[georgia_match], ambiguous=["fragile"])
        classifier = FakeClassifier(
            clarification_required=True,
            ambiguous_terms=[
                AmbiguousTerm(phrase="fragile", clarification_question="What?")
            ],
        )
        store = InMemoryClarificationStore()
        input_func, output_func, _ = _fake_io(["battery recycling focus"])
        prompter = TerminalClarificationPrompter(input_func=input_func, output_func=output_func)
        resolver = ClarificationResolver(
            store=store, query_analyzer=analyzer, phrase_classifier=classifier
        )
        workflow = TerminalClarificationWorkflow(
            analyzer=analyzer,
            phrase_classifier=classifier,
            store=store,
            prompter=prompter,
            resolver=resolver,
        )

        ctx = workflow.analyze_and_maybe_clarify("Show fragile suppliers in Georgia")
        canonical_values = [m.canonical_value for m in ctx.merged_matched_vocabulary]
        assert "Georgia" in canonical_values
        assert "Battery recycling" in canonical_values
        assert any("clarified" in note.lower() for note in ctx.final_generation_notes)


class TestDoubleResolveRaises:
    def test_double_resolve_raises(self) -> None:
        analyzer = FakeAnalyzer(ambiguous=["fragile"])
        classifier = FakeClassifier(
            clarification_required=True,
            ambiguous_terms=[
                AmbiguousTerm(phrase="fragile", clarification_question="What?")
            ],
        )
        store = InMemoryClarificationStore()
        input_func1, output_func1, _ = _fake_io(["means low employment"])
        prompter1 = TerminalClarificationPrompter(input_func=input_func1, output_func=output_func1)
        resolver = ClarificationResolver(
            store=store, query_analyzer=analyzer, phrase_classifier=classifier
        )
        workflow = TerminalClarificationWorkflow(
            analyzer=analyzer,
            phrase_classifier=classifier,
            store=store,
            prompter=prompter1,
            resolver=resolver,
        )

        ctx = workflow.analyze_and_maybe_clarify("Show fragile suppliers")
        assert len(ctx.resolved_clarifications) == 1

        # Trying to resolve again with the same ID should raise.
        from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator.models import (
            ClarificationAnswer,
            ClarificationSubmission,
        )
        # Find the clarification_id from the store.
        sessions = list(store._sessions.values())
        assert len(sessions) == 1
        cid = sessions[0].clarification_id
        with pytest.raises(ClarificationAlreadyResolvedError):
            resolver.resolve(
                ClarificationSubmission(
                    clarification_id=cid,
                    answers=[ClarificationAnswer(term="fragile", custom_text="again")],
                )
            )
