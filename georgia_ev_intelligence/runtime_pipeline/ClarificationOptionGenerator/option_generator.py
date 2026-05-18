"""Generate clarification requests from query analysis results."""
from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, Protocol

from ..query_analyzer.models import QueryAnalysisResult
from .constants import (
    DEFAULT_UNKNOWN_TERM_OPTIONS,
    QUESTION_TEMPLATE_KNOWN,
    QUESTION_TEMPLATE_UNKNOWN,
    STATUS_PENDING,
)
from .concept_registry import ConceptRegistryProtocol, normalize_term
from .clarification_store import ClarificationStoreProtocol
from .models import (
    AmbiguousTermClarification,
    ClarificationRequest,
    StoredClarificationSession,
    utc_now,
)


class ClarificationOptionGenerator:
    """Build clarification requests for ambiguous query terms."""

    def __init__(
        self,
        concept_registry: ConceptRegistryProtocol,
        store: ClarificationStoreProtocol | None = None,
        id_generator: Callable[[], str] | None = None,
    ) -> None:
        self._registry = concept_registry
        self._store = store
        self._id_generator = id_generator or (lambda: str(uuid.uuid4()))

    def generate_request(
        self,
        original_query: str,
        analysis: QueryAnalysisResult | Any,
    ) -> ClarificationRequest | None:
        ambiguous_terms = _get_ambiguous_terms(analysis)
        if not ambiguous_terms:
            return None

        clarification_id = self._id_generator()
        term_clarifications: list[AmbiguousTermClarification] = []

        for term in ambiguous_terms:
            normalized = normalize_term(term)
            concept = self._registry.find_concept(term)
            if concept:
                options = list(concept.clarification_options)
                question = QUESTION_TEMPLATE_KNOWN.format(term=term)
                matched_concept_id = concept.concept_id
            else:
                options = list(DEFAULT_UNKNOWN_TERM_OPTIONS)
                question = QUESTION_TEMPLATE_UNKNOWN.format(term=term)
                matched_concept_id = None

            term_clarifications.append(
                AmbiguousTermClarification(
                    term=term,
                    normalized_term=normalized,
                    matched_concept_id=matched_concept_id,
                    question=question,
                    options=options,
                )
            )

        request = ClarificationRequest(
            clarification_required=True,
            clarification_id=clarification_id,
            original_query=original_query,
            ambiguous_terms=term_clarifications,
            status=STATUS_PENDING,
            created_at=utc_now(),
        )

        if self._store is not None:
            session = StoredClarificationSession(
                clarification_id=clarification_id,
                original_query=original_query,
                original_analysis=analysis,
                request=request,
                status=STATUS_PENDING,
            )
            self._store.save_session(session)

        return request


def _get_ambiguous_terms(analysis: Any) -> list[str]:
    if isinstance(analysis, QueryAnalysisResult):
        return list(analysis.ambiguous_terms)
    terms = getattr(analysis, "ambiguous_terms", None)
    if terms is None:
        return []
    return list(terms)
