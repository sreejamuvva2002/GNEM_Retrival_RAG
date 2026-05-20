"""Configuration for the isolated hybrid retrieval module."""
from __future__ import annotations

import os
from dataclasses import dataclass


RETRIEVER_TOP_K = int(os.environ.get("HYBRID_RETRIEVER_TOP_K", "250"))
RERANKER_TOP_K = int(os.environ.get("HYBRID_RERANKER_TOP_K", "45"))
RERANKER_MODEL = os.environ.get(
    "HYBRID_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L12-v2",
)


@dataclass(frozen=True)
class HybridRetrievalConfig:
    """Runtime knobs for child retrieval and reranking."""

    retriever_top_k: int = RETRIEVER_TOP_K
    reranker_top_k: int = RERANKER_TOP_K
    reranker_model: str = RERANKER_MODEL
