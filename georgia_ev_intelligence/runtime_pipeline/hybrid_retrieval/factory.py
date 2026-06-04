"""Factory for the default hybrid retrieval pipeline.

WHY THIS FILE EXISTS
--------------------
Provides a single entry point — ``build_default_pipeline()`` — for wiring up
all the hybrid retrieval components (BM25 retriever, dense retriever, merger,
parent mapper, cross-encoder reranker) into a ready-to-use
``HybridRetrievalOrchestrator``.

This factory pattern hides the dependency wiring from callers, making it easy
to construct the pipeline with one function call while keeping each component
independently testable via constructor injection.

WHAT IS WIRED TOGETHER
-----------------------
- ``BM25ChildRetriever``     — sparse keyword search over in-memory BM25 index
- ``DenseChildRetriever``    — vector cosine search via pgvector
- ``CrossEncoderReranker``   — parent-level cross-encoder reranking
- ``ChildResultMerger``      — merge + deduplicate child hits by chunk_id
- ``ParentChildMapper``      — map child hits → parent records (batch SQL fetch)

All defaults come from ``HybridRetrievalConfig`` (overridable via env vars).
Use this factory in ``run_baseline.py`` to create the retrieval cache.
"""
from __future__ import annotations

from .bm25_retriever import BM25ChildRetriever
from .config import HybridRetrievalConfig
from .dense_retriever import DenseChildRetriever
from .merger import ChildResultMerger
from .orchestrator import HybridRetrievalOrchestrator
from .parent_mapper import ParentChildMapper
from .reranker import CrossEncoderReranker


def build_default_pipeline(
    config: HybridRetrievalConfig | None = None,
) -> HybridRetrievalOrchestrator:
    """Build the default BM25 + dense + cross-encoder retrieval pipeline."""
    cfg = config or HybridRetrievalConfig()
    return HybridRetrievalOrchestrator(
        retrievers=(
            BM25ChildRetriever(),
            DenseChildRetriever(),
        ),
        reranker=CrossEncoderReranker(model_name=cfg.reranker_model),
        merger=ChildResultMerger(),
        parent_mapper=ParentChildMapper(),
        config=cfg,
    )
