"""Semantic retriever factory used by all vector-search call sites."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from ...shared import config
from .dense import DenseRetriever
from ...shared.data.schema import SKIP_COLUMNS


class SemanticRetriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 15,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        ...


def build_semantic_retriever(df: pd.DataFrame) -> SemanticRetriever:
    """
    Build the one semantic/vector retriever used by the pipeline.

    Uses pgvector (Neon PostgreSQL) by default. The in-memory DenseRetriever
    is retained as an explicit local-development fallback when
    USE_PGVECTOR_RETRIEVER=false.
    """
    if config.USE_PGVECTOR_RETRIEVER:
        from .pgvector import PgVectorRetriever

        return PgVectorRetriever(
            df=df,
            model_name=config.EMBEDDING_MODEL,
        )

    return DenseRetriever(
        df=df,
        model_name=config.EMBEDDING_MODEL,
        skip_cols=SKIP_COLUMNS,
    )


def retriever_backend_label() -> str:
    if config.USE_PGVECTOR_RETRIEVER:
        return f"pgvector/{config.EMBEDDING_MODEL}"
    return f"InMemory/{config.EMBEDDING_MODEL}"
