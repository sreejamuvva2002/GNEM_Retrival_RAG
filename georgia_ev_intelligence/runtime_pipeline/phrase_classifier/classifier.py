"""Orchestrates LLM remaining phrase classification."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..query_analyzer.models import QueryAnalysisResult
from .llm_client import call_classifier_llm
from .models import PhraseClassificationResult
from .parser import parse_response
from .prompt import build_prompt

logger = logging.getLogger(__name__)


class RemainingPhraseClassifier:
    """Classify leftover unmatched phrases using an LLM."""

    def __init__(
        self,
        llm_caller: Callable[[str, int | None], str] | None = None,
        prompt_builder: Callable[..., str] | None = None,
        response_parser: Callable[[str, list[str]], PhraseClassificationResult] | None = None,
    ) -> None:
        self._call_llm = llm_caller or call_classifier_llm
        self._build_prompt = prompt_builder or build_prompt
        self._parse_response = response_parser or parse_response

    def classify(
        self,
        original_query: str,
        analysis: QueryAnalysisResult,
    ) -> PhraseClassificationResult:
        """Classify remaining unmatched phrases from a query analysis.

        Returns a PhraseClassificationResult. Never raises — falls back to
        conservative classification on any failure.
        """
        phrases = list(analysis.ambiguous_terms)

        if not phrases:
            return PhraseClassificationResult(
                clarification_required=False,
                debug={"skipped": True, "reason": "no_unmatched_phrases"},
            )

        matched_display = _format_matched_vocabulary(analysis)
        operation = analysis.operation or "(none)"
        target_entity = analysis.target_entity or "(none)"
        phrases_display = "\n".join(f"- {p}" for p in phrases)

        prompt = self._build_prompt(
            original_query=original_query,
            matched_vocabulary_terms=matched_display,
            detected_operation=operation,
            detected_target_entity=target_entity,
            remaining_unmatched_phrases=phrases_display,
        )

        raw_response = self._call_llm(prompt, None)
        result = self._parse_response(raw_response, phrases)
        return result


def _format_matched_vocabulary(analysis: QueryAnalysisResult) -> str:
    """Format matched vocabulary for the prompt."""
    if not analysis.matched_vocabulary:
        return "(none)"
    lines: list[str] = []
    for m in analysis.matched_vocabulary:
        lines.append(
            f"- \"{m.matched_text}\" → canonical: \"{m.canonical_value}\", "
            f"column: \"{m.source_column}\", type: \"{m.term_type or 'unknown'}\", "
            f"match: {m.match_type}"
        )
    return "\n".join(lines)
