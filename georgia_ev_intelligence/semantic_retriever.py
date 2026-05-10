"""Semantic retriever factory used by all vector-search call sites."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from . import config
from .dense_retriever import DenseRetriever
from .schema_index import SKIP_COLUMNS


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

    With the current project configuration this is the parent-child Qdrant
    retriever backed by Nomic embeddings. The in-memory retriever is retained
    only as an explicit local-development fallback.
    """
    if config.USE_QDRANT_RETRIEVER:
        from .qdrant_retriever import QdrantRetriever

        return QdrantRetriever(
            df=df,
            model_name=config.EMBEDDING_MODEL,
            collection_name=config.QDRANT_COLLECTION,
        )

    return DenseRetriever(
        df=df,
        model_name=config.EMBEDDING_MODEL,
        skip_cols=SKIP_COLUMNS,
    )


def retriever_backend_label() -> str:
    if config.USE_QDRANT_RETRIEVER:
        return f"Qdrant/{config.EMBEDDING_MODEL}"
    return f"InMemory/{config.EMBEDDING_MODEL}"
