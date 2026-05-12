"""Embedding model helpers shared by in-memory and Qdrant retrieval."""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

from .. import config


def load_sentence_transformer(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(
        model_name,
        trust_remote_code=config.EMBEDDING_TRUST_REMOTE_CODE,
        local_files_only=config.EMBEDDING_LOCAL_FILES_ONLY,
    )


def as_document_text(text: str) -> str:
    return _prefix(config.EMBEDDING_DOCUMENT_PREFIX, text)


def as_query_text(text: str) -> str:
    return _prefix(config.EMBEDDING_QUERY_PREFIX, text)


def _prefix(prefix: str, text: str) -> str:
    clean = str(text or "").strip()
    if not prefix:
        return clean

    normalized_prefix = prefix if prefix.endswith(" ") else f"{prefix} "
    if clean.lower().startswith(normalized_prefix.lower()):
        return clean
    return f"{normalized_prefix}{clean}"
