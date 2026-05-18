"""Resolve user clarification submissions into structured context."""
from __future__ import annotations

from typing import Any, Protocol

from ..query_analyzer.models import QueryAnalysisResult
from ..phrase_classifier.classifier import RemainingPhraseClassifier
from .analysis_merger import AnalysisMerger
from .clarification_store import ClarificationStoreProtocol
from .exceptions import ClarificationAlreadyResolvedError
from .models import (
    ClarificationSubmission,
    ResolvedClarification,
    ResolvedQueryContext,
    utc_now,
)

_STATUS_PENDING = "pending"
_STATUS_RESOLVED = "resolved"


class QueryAnalyzerProtocol(Protocol):
    def analyze(self, query: str) -> QueryAnalysisResult:
        ...


class ClarificationResolver:
    """Validate answers and produce ResolvedQueryContext."""

    def __init__(
        self,
        store: ClarificationStoreProtocol,
        query_analyzer: QueryAnalyzerProtocol | None = None,
        phrase_classifier: RemainingPhraseClassifier | None = None,
        merger: AnalysisMerger | None = None,
    ) -> None:
        self._store = store
        self._query_analyzer = query_analyzer
        self._phrase_classifier = phrase_classifier
        self._merger = merger or AnalysisMerger()

    def resolve(self, submission: ClarificationSubmission) -> ResolvedQueryContext:
        session = self._store.get_session(submission.clarification_id)
        if session.status != _STATUS_PENDING:
            raise ClarificationAlreadyResolvedError(
                f"Clarification session {submission.clarification_id} is not pending."
            )

        # Build resolved clarifications from user answers.
        resolved: list[ResolvedClarification] = []
        for answer in submission.answers:
            text = answer.custom_text.strip() if answer.custom_text else ""
            if not text:
                resolved.append(
                    ResolvedClarification(term=answer.term, meaning="", ignored=True)
                )
            else:
                resolved.append(
                    ResolvedClarification(term=answer.term, meaning=text, ignored=False)
                )

        # Build combined clarification text from non-ignored answers.
        clarification_text = _build_clarification_text(resolved)

        # Re-analyze clarification text if analyzer is available.
        clarification_analysis = None
        clarification_phrase_classification = None
        if self._query_analyzer is not None and clarification_text.strip():
            clarification_analysis = self._query_analyzer.analyze(clarification_text)
            # Re-classify clarification leftovers — do NOT trigger another clarification loop.
            if self._phrase_classifier is not None and clarification_analysis.ambiguous_terms:
                clarification_phrase_classification = self._phrase_classifier.classify(
                    clarification_text, clarification_analysis
                )

        # Merge everything.
        context = self._merger.merge(
            original_query=session.original_query,
            original_analysis=session.original_analysis,
            original_phrase_classification=session.phrase_classification,
            resolved_clarifications=resolved,
            clarification_analysis=clarification_analysis,
            clarification_phrase_classification=clarification_phrase_classification,
        )

        # Mark session resolved.
        session.status = _STATUS_RESOLVED
        session.resolved_context = context
        session.updated_at = utc_now()
        self._store.update_session(session)

        return context


def _build_clarification_text(
    resolved_clarifications: list[ResolvedClarification],
) -> str:
    parts: list[str] = []
    for rc in resolved_clarifications:
        if rc.ignored or not rc.meaning:
            continue
        parts.append(rc.meaning.strip())
    return " ".join(parts)
