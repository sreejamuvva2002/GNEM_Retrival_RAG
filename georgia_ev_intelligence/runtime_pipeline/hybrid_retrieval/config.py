"""Configuration for the isolated hybrid retrieval module."""
from __future__ import annotations

from dataclasses import dataclass


RETRIEVER_TOP_K = 100
RERANKER_TOP_K = 45
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class HybridRetrievalConfig:
    """Runtime knobs for child retrieval and reranking."""

    retriever_top_k: int = RETRIEVER_TOP_K
    reranker_top_k: int = RERANKER_TOP_K
    reranker_model: str = RERANKER_MODEL
