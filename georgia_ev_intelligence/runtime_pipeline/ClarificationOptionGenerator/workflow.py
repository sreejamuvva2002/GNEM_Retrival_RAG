"""Orchestrate terminal clarification prompting and resolution."""
from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Protocol

from ..query_analyzer.models import QueryAnalysisResult
from ..phrase_classifier.classifier import RemainingPhraseClassifier
from ..phrase_classifier.models import PhraseClassificationResult
from .analysis_merger import AnalysisMerger
from .clarification_resolver import ClarificationResolver
from .clarification_store import ClarificationStoreProtocol
from .models import (
    ClarificationQuestion,
    ClarificationRequest,
    ResolvedQueryContext,
    StoredClarificationSession,
    utc_now,
)
from .terminal_prompter import ClarificationPrompterProtocol

_STATUS_PENDING = "pending"


class QueryAnalyzerProtocol(Protocol):
    def analyze(self, query: str) -> QueryAnalysisResult:
        ...


class TerminalClarificationWorkflow:
    """Run analyze -> classify -> clarify (if needed) -> resolve for terminal use."""

    def __init__(
        self,
        analyzer: QueryAnalyzerProtocol,
        phrase_classifier: RemainingPhraseClassifier,
        store: ClarificationStoreProtocol,
        prompter: ClarificationPrompterProtocol,
        resolver: ClarificationResolver,
        id_generator: Callable[[], str] | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._phrase_classifier = phrase_classifier
        self._store = store
        self._prompter = prompter
        self._resolver = resolver
        self._id_generator = id_generator or (lambda: str(uuid.uuid4()))
        self._merger = AnalysisMerger()

    def analyze_and_maybe_clarify(self, query: str) -> ResolvedQueryContext:
        """Analyze a query, classify leftovers, and run clarification when needed.

        When no clarification is required, returns a ResolvedQueryContext built
        directly from the analysis and phrase classification.
        """
        analysis = self._analyzer.analyze(query)
        phrase_result = self._phrase_classifier.classify(query, analysis)

        if not phrase_result.clarification_required:
            return self._build_no_clarification_context(
                query, analysis, phrase_result
            )

        # Build clarification request from ambiguous terms.
        clarification_id = self._id_generator()
        questions = [
            ClarificationQuestion(term=at.phrase, question=at.clarification_question)
            for at in phrase_result.ambiguous_terms
        ]
        request = ClarificationRequest(
            clarification_required=True,
            clarification_id=clarification_id,
            original_query=query,
            questions=questions,
            status=_STATUS_PENDING,
            created_at=utc_now(),
        )

        # Store session for resolver to find later.
        session = StoredClarificationSession(
            clarification_id=clarification_id,
            original_query=query,
            original_analysis=analysis,
            phrase_classification=phrase_result,
            request=request,
            status=_STATUS_PENDING,
        )
        self._store.save_session(session)

        # Prompt user and resolve.
        submission = self._prompter.prompt(request)
        return self._resolver.resolve(submission)

    def _build_no_clarification_context(
        self,
        query: str,
        analysis: QueryAnalysisResult,
        phrase_result: PhraseClassificationResult,
    ) -> ResolvedQueryContext:
        """Build ResolvedQueryContext when no clarification is needed."""
        return self._merger.merge(
            original_query=query,
            original_analysis=analysis,
            original_phrase_classification=phrase_result,
            resolved_clarifications=[],
            clarification_analysis=None,
            clarification_phrase_classification=None,
        )
