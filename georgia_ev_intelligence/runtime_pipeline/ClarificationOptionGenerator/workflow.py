"""Orchestrate terminal clarification prompting and resolution."""
from __future__ import annotations

from typing import Protocol

from ..query_analyzer.models import QueryAnalysisResult
from .analysis_merger import AnalysisMerger
from .clarification_resolver import ClarificationResolver, QueryAnalyzerProtocol
from .io_protocols import ClarificationPrompterProtocol
from .models import ResolvedQueryContext
from .option_generator import ClarificationOptionGenerator


class TerminalClarificationWorkflow:
    """Run analyze → clarify (if needed) → resolve for terminal use."""

    def __init__(
        self,
        analyzer: QueryAnalyzerProtocol,
        option_generator: ClarificationOptionGenerator,
        prompter: ClarificationPrompterProtocol,
        resolver: ClarificationResolver,
        merger: AnalysisMerger | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._option_generator = option_generator
        self._prompter = prompter
        self._resolver = resolver
        self._merger = merger or AnalysisMerger()

    def handle_query(self, query: str) -> ResolvedQueryContext:
        """Analyze a query and run clarification when ambiguous terms exist.

        When no clarification is required, returns a ResolvedQueryContext built
        directly from the original analysis (no prompting, clarification_id="").
        """
        analysis = self._analyzer.analyze(query)
        request = self._option_generator.generate_request(query, analysis)

        if request is None:
            return self._build_no_clarification_context(query, analysis)

        submission = self._prompter.prompt(request)
        return self._resolver.resolve(submission)

    def _build_no_clarification_context(
        self,
        query: str,
        analysis: QueryAnalysisResult,
    ) -> ResolvedQueryContext:
        return self._merger.merge(
            original_query=query,
            original_analysis=analysis,
            resolved_clarifications=[],
            clarification_analysis=None,
            clarification_id="",
        )
