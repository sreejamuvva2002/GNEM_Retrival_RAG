"""Active three-stage hybrid retrieval module."""
from __future__ import annotations

from .bm25_retriever import BM25ChildRetriever
from .config import (
    RERANKER_MODEL,
    RERANKER_TOP_K,
    RETRIEVER_TOP_K,
    HybridRetrievalConfig,
)
from .dense_retriever import DenseChildRetriever
from .factory import build_default_pipeline
from .interfaces import ChildRetriever, ParentReranker
from .merger import ChildResultMerger
from .models import HybridRetrievalResult, HybridRetrievalTrace, RerankedChildChunk
from .orchestrator import HybridRetrievalOrchestrator
from .parent_mapper import ParentChildMapper
from .reranker import CrossEncoderReranker

__all__ = [
    "BM25ChildRetriever",
    "ChildRetriever",
    "ChildResultMerger",
    "CrossEncoderReranker",
    "DenseChildRetriever",
    "HybridRetrievalConfig",
    "HybridRetrievalResult",
    "HybridRetrievalTrace",
    "HybridRetrievalOrchestrator",
    "ParentChildMapper",
    "ParentReranker",
    "RERANKER_MODEL",
    "RERANKER_TOP_K",
    "RETRIEVER_TOP_K",
    "RerankedChildChunk",
    "build_default_pipeline",
]
