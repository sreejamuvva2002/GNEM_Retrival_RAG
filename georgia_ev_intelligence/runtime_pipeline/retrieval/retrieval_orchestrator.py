"""Retrieval orchestrator for hybrid dense + BM25 retrieval."""
from __future__ import annotations

from dataclasses import dataclass

from ..schemas import (
    FusedChildChunk,
    ParentContext,
    PipelineConfig,
    RetrievedChildChunk,
)
from .hybrid_retriever import HybridRetriever
from .parent_fetcher import fetch_parents


@dataclass
class OrchestratorResult:
    """Result from hybrid retrieval and parent fetch."""

    parent_contexts: list[ParentContext]
    fused_children: list[FusedChildChunk]
    dense_results: list[RetrievedChildChunk]
    bm25_results: list[RetrievedChildChunk]


class RetrievalOrchestrator:
    """Hybrid retrieval over child chunks, then parent fetch."""

    def __init__(self, pipeline_config: PipelineConfig | None = None):
        self._config = pipeline_config or PipelineConfig()
        self._hybrid_retriever = HybridRetriever(pipeline_config=self._config)

    def search(self, question: str) -> OrchestratorResult:
        fused_children, dense_results, bm25_results = self._hybrid_retriever.search(
            question
        )
        parent_contexts = fetch_parents(
            fused_children,
            dense_results,
            bm25_results,
            pipeline_config=self._config,
        )
        return OrchestratorResult(
            parent_contexts=parent_contexts,
            fused_children=fused_children,
            dense_results=dense_results,
            bm25_results=bm25_results,
        )
