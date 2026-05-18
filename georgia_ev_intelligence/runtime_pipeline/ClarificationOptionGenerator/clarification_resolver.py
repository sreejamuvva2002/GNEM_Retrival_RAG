"""Resolve user clarification submissions into structured context."""
from __future__ import annotations

from typing import Any, Protocol

from ..query_analyzer.models import QueryAnalysisResult
from .analysis_merger import AnalysisMerger
from .clarification_store import ClarificationStoreProtocol
from .concept_registry import normalize_term
from .constants import STATUS_PENDING, STATUS_RESOLVED
from .exceptions import (
    ClarificationAlreadyResolvedError,
    InvalidClarificationAnswerError,
    MissingCustomClarificationTextError,
)
from .models import (
    AmbiguousTermClarification,
    ClarificationAnswer,
    ClarificationOption,
    ClarificationSubmission,
    ResolvedClarification,
    ResolvedQueryContext,
    utc_now,
)


class QueryAnalyzerProtocol(Protocol):
    def analyze(self, query: str) -> QueryAnalysisResult:
        ...


class ClarificationResolver:
    """Validate answers and produce ResolvedQueryContext."""

    def __init__(
        self,
        store: ClarificationStoreProtocol,
        query_analyzer: QueryAnalyzerProtocol | None = None,
        merger: AnalysisMerger | None = None,
    ) -> None:
        self._store = store
        self._query_analyzer = query_analyzer
        self._merger = merger or AnalysisMerger()

    def resolve(self, submission: ClarificationSubmission) -> ResolvedQueryContext:
        session = self._store.get_session(submission.clarification_id)
        if session.status != STATUS_PENDING:
            raise ClarificationAlreadyResolvedError(
                f"Clarification session {submission.clarification_id} is not pending."
            )

        term_map = {
            normalize_term(at.term): at for at in session.request.ambiguous_terms
        }
        resolved: list[ResolvedClarification] = []

        for answer in submission.answers:
            resolved.append(
                _resolve_single_answer(answer, term_map)
            )

        clarification_text = _build_clarification_text(resolved)
        clarification_analysis = None
        if self._query_analyzer is not None and clarification_text.strip():
            clarification_analysis = self._query_analyzer.analyze(
                clarification_text
            )

        context = self._merger.merge(
            original_query=session.original_query,
            original_analysis=session.original_analysis,
            resolved_clarifications=resolved,
            clarification_analysis=clarification_analysis,
            clarification_id=session.clarification_id,
        )

        session.status = STATUS_RESOLVED
        session.resolved_context = context
        session.updated_at = utc_now()
        self._store.update_session(session)

        return context


def _resolve_single_answer(
    answer: ClarificationAnswer,
    term_map: dict[str, AmbiguousTermClarification],
) -> ResolvedClarification:
    key = normalize_term(answer.term)
    term_entry = term_map.get(key)
    if term_entry is None:
        raise InvalidClarificationAnswerError(
            f"Unknown ambiguous term in answer: {answer.term!r}"
        )

    option_id = answer.selected_option_id
    custom_text = (answer.custom_text or "").strip() or None

    if option_id is None and custom_text:
        option_id = "custom"

    if option_id is None:
        raise InvalidClarificationAnswerError(
            f"No option selected for term: {answer.term!r}"
        )

    option = _find_option(term_entry.options, option_id)
    if option is None:
        raise InvalidClarificationAnswerError(
            f"Invalid option id {option_id!r} for term {answer.term!r}"
        )

    if option.ignore_term:
        return ResolvedClarification(
            term=term_entry.term,
            selected_option_id=option.id,
            meaning=None,
            ignored=True,
            custom_text=None,
            matched_concept_id=term_entry.matched_concept_id,
        )

    if option.requires_custom_text or option.id == "custom":
        if not custom_text:
            raise MissingCustomClarificationTextError(
                f"Custom text required for term: {answer.term!r}"
            )
        return ResolvedClarification(
            term=term_entry.term,
            selected_option_id=option.id,
            meaning=custom_text,
            ignored=False,
            custom_text=custom_text,
            matched_concept_id=term_entry.matched_concept_id,
        )

    return ResolvedClarification(
        term=term_entry.term,
        selected_option_id=option.id,
        meaning=option.meaning,
        ignored=False,
        custom_text=None,
        matched_concept_id=term_entry.matched_concept_id,
    )


def _find_option(
    options: list[ClarificationOption],
    option_id: str,
) -> ClarificationOption | None:
    for opt in options:
        if opt.id == option_id:
            return opt
    return None


def _build_clarification_text(
    resolved_clarifications: list[ResolvedClarification],
) -> str:
    parts: list[str] = []
    for rc in resolved_clarifications:
        if rc.ignored or not rc.meaning:
            continue
        parts.append(rc.meaning.strip())
    return " ".join(parts)
