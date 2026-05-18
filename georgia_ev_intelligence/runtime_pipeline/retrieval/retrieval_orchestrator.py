"""Retrieval orchestrator merging hybrid and vocabulary filter tracks.

Single entry point for all retrieval in the runtime pipeline.
Orchestrates two parallel retrieval tracks and merges their results:
  Track A (Hybrid): Dense + BM25 -> RRF fusion -> parent fetch
  Track B (Vocabulary Filter): query rewrite -> vocab match -> exact parent fetch
Track B results are prioritised in the final merge.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field

from ...shared import config
from ..query_rewriting.models import StructuredQuery, VocabularyMatches
from ..query_rewriting.rewriter import QueryRewriter
from ..query_rewriting.vocabulary_matcher import VocabularyMatcher
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.parent_fetcher import fetch_parents
from ..retrieval.vocabulary_filter_retriever import VocabularyFilterRetriever
from ..schemas import (
    FusedChildChunk,
    ParentContext,
    PipelineConfig,
    RetrievedChildChunk,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Complete result from the retrieval orchestrator.

    Carries all intermediate results for tracing and downstream use.
    """

    parent_contexts: list[ParentContext]
    fused_children: list[FusedChildChunk]
    dense_results: list[RetrievedChildChunk]
    bm25_results: list[RetrievedChildChunk]
    vocabulary_parents: list[ParentContext]
    structured_query: StructuredQuery
    vocabulary_matches: VocabularyMatches
    vocabulary_used: bool = False


class RetrievalOrchestrator:
    """Orchestrates two parallel retrieval tracks and merges results.

    Track A (Hybrid): Dense + BM25 -> RRF fusion -> parent fetch
    Track B (Vocabulary Filter): query rewrite -> vocab match
                                 -> exact parent fetch
    Track B results are prioritised in the final merge.
    """

    def __init__(self, pipeline_config: PipelineConfig | None = None):
        """Initialise all sub-components."""
        self._config = pipeline_config or PipelineConfig()
        self._query_rewriter = QueryRewriter()
        self._vocabulary_matcher = VocabularyMatcher()
        self._hybrid_retriever = HybridRetriever(pipeline_config=self._config)
        self._vocabulary_filter_retriever = VocabularyFilterRetriever()

    def search(self, question: str) -> OrchestratorResult:
        """Execute full retrieval flow across both tracks.

        Step 1: Query Rewriting (StructuredQuery from LLM)
        Step 2: Vocabulary Matching (resolve terms to row_ids)
        Step 3: Track A - Hybrid Retrieval (always runs)
        Step 4: Track B - Vocabulary Filter (only if has_matches)
        Step 5: Merge results from both tracks
        """
        # Step 1: Query Rewriting
        structured_query = self._query_rewriter.rewrite(question)

        # Step 2: Vocabulary Matching
        if structured_query.has_filters():
            vocabulary_matches = self._vocabulary_matcher.match(structured_query)
        else:
            vocabulary_matches = VocabularyMatches(
                structured_query=structured_query
            )

        # Step 3: Track A - Hybrid Retrieval (always runs)
        fused_children, dense_results, bm25_results = (
            self._hybrid_retriever.search(question)
        )
        hybrid_parents = fetch_parents(
            fused_children, dense_results, bm25_results, pipeline_config=self._config
        )

        # Step 4: Track B - Vocabulary Filter (only if has_matches)
        vocabulary_parents: list[ParentContext] = []
        if vocabulary_matches.has_matches:
            vocabulary_parents = self._vocabulary_filter_retriever.search(
                vocabulary_matches
            )

        # Step 5: Merge
        parent_contexts = self._merge(hybrid_parents, vocabulary_parents)

        vocabulary_used = (
            vocabulary_matches.has_matches and len(vocabulary_parents) > 0
        )

        return OrchestratorResult(
            parent_contexts=parent_contexts,
            fused_children=fused_children,
            dense_results=dense_results,
            bm25_results=bm25_results,
            vocabulary_parents=vocabulary_parents,
            structured_query=structured_query,
            vocabulary_matches=vocabulary_matches,
            vocabulary_used=vocabulary_used,
        )

    def _merge(
        self,
        hybrid_parents: list[ParentContext],
        vocabulary_parents: list[ParentContext],
    ) -> list[ParentContext]:
        """Merge results from both tracks.

        Strategy:
        1. Build a dict keyed by record_id.
        2. Insert all vocabulary_parents first (they have higher scores).
        3. For each hybrid_parent:
           - If record_id already in dict (found by both tracks):
             combined_score = max(existing_score, hybrid_score) * 1.1
             (10% bonus for appearing in both tracks)
           - If not in dict: add as-is.
        4. Sort by combined_score descending.
        5. Return top config.parent_top_k results.
        """
        if not vocabulary_parents:
            return hybrid_parents

        merged: dict[str, ParentContext] = {}

        # Insert vocabulary parents first (higher base score)
        for vp in vocabulary_parents:
            merged[vp.record_id] = vp

        # Merge hybrid parents
        for hp in hybrid_parents:
            if hp.record_id in merged:
                # Found by both tracks: apply 10% overlap bonus
                existing = merged[hp.record_id]
                boosted_score = max(existing.combined_score, hp.combined_score) * 1.1
                merged[hp.record_id] = dataclasses.replace(
                    existing, combined_score=boosted_score
                )
            else:
                merged[hp.record_id] = hp

        # Sort by combined_score descending and take top_k
        sorted_parents = sorted(
            merged.values(), key=lambda p: p.combined_score, reverse=True
        )
        return sorted_parents[: self._config.parent_top_k]
