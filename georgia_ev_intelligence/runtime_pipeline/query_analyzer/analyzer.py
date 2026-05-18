"""Orchestrates deterministic query analysis."""
from __future__ import annotations

from .ambiguous_extractor import AmbiguousTermExtractor
from .intent_detector import IntentDetector
from .models import QueryAnalysisResult
from .normalizer import QueryNormalizer
from .span_matcher import LongestSpanMatcher
from .vocabulary_index import VocabularyIndex
from .vocabulary_repository import VocabularyRepository


class QueryAnalyzer:
    """Analyze a raw user query against vocabulary terms."""

    def __init__(
        self,
        vocabulary_repository: VocabularyRepository,
        normalizer: QueryNormalizer | None = None,
        matcher: LongestSpanMatcher | None = None,
        intent_detector: IntentDetector | None = None,
        ambiguous_extractor: AmbiguousTermExtractor | None = None,
    ) -> None:
        self._repository = vocabulary_repository
        self._normalizer = normalizer or QueryNormalizer()
        self._matcher = matcher or LongestSpanMatcher()
        self._intent_detector = intent_detector or IntentDetector(self._normalizer)
        self._ambiguous_extractor = ambiguous_extractor or AmbiguousTermExtractor()
        self._index: VocabularyIndex | None = None

    def _get_index(self) -> VocabularyIndex:
        if self._index is None:
            self._index = VocabularyIndex(self._repository.load_terms())
        return self._index

    def analyze(self, query: str) -> QueryAnalysisResult:
        normalized = self._normalizer.normalize(query)
        tokens = normalized.tokens
        index = self._get_index()

        span_result = self._matcher.match(tokens, index)
        intent = self._intent_detector.detect(normalized.normalized, tokens)

        excluded = (
            span_result.occupied_token_indices
            | intent.operation_token_indices
            | intent.target_entity_token_indices
        )
        ambiguous_terms, ignored_from_remainder = self._ambiguous_extractor.extract(
            tokens, frozenset(excluded)
        )

        ignored_tokens = list(dict.fromkeys(ignored_from_remainder))
        unmatched_tokens = [tokens[i] for i in span_result.unmatched_token_indices]

        return QueryAnalysisResult(
            original_query=normalized.original,
            normalized_query=normalized.normalized,
            operation=intent.operation,
            target_entity=intent.target_entity,
            matched_vocabulary=span_result.matches,
            ambiguous_terms=ambiguous_terms,
            ignored_tokens=ignored_tokens,
            unmatched_tokens=unmatched_tokens,
            debug={
                "tokens": list(tokens),
                "occupied_token_indices": sorted(span_result.occupied_token_indices),
                "unmatched_token_indices": list(span_result.unmatched_token_indices),
                "operation_token_indices": sorted(intent.operation_token_indices),
                "target_entity_token_indices": sorted(intent.target_entity_token_indices),
                "vocabulary_phrase_count": index.phrase_count,
                "max_phrase_length": index.max_phrase_length,
                "match_count": len(span_result.matches),
            },
        )
