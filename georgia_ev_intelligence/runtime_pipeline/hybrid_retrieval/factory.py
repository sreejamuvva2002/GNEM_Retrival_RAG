"""Factory for the default isolated hybrid retrieval pipeline."""
from __future__ import annotations

from .bm25_retriever import BM25ChildRetriever
from .config import HybridRetrievalConfig
from .dense_retriever import DenseChildRetriever
from .interfaces import RetrieverStage
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
            RetrieverStage(name="bm25", retriever=BM25ChildRetriever()),
            RetrieverStage(name="dense", retriever=DenseChildRetriever()),
        ),
        reranker=CrossEncoderReranker(model_name=cfg.reranker_model),
        merger=ChildResultMerger(),
        parent_mapper=ParentChildMapper(),
        config=cfg,
    )
