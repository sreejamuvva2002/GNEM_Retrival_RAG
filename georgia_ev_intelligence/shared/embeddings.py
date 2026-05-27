"""Embedding model helpers shared by the offline indexer and online dense retriever.

WHY THIS FILE EXISTS:
  Provides a single place to load the SentenceTransformer model and apply the
  asymmetric prefixes that nomic-embed-text requires.  The model uses
  "search_document:" for KB records at index time and "search_query:" for user
  queries at runtime — without those prefixes cosine similarity degrades
  significantly.

TECHNIQUE:
  Asymmetric embedding with nomic-ai/nomic-embed-text-v1.5.
  The document prefix is appended when indexing child chunks (offline).
  The query prefix is appended when embedding a user question (online).
  Both prefixes are read from environment variables (EMBEDDING_DOCUMENT_PREFIX
  / EMBEDDING_QUERY_PREFIX) so they can be changed without code edits.

FUNCTIONS:
  load_sentence_transformer(model_name) — loads the model from HuggingFace or
      local cache (controlled by EMBEDDING_LOCAL_FILES_ONLY).
  as_document_text(text)  — prepends document prefix (used at index time).
  as_query_text(text)     — prepends query prefix (used at retrieval time).

RELATIONSHIPS:
  Used by: offline_pipeline/pgvector_store.py (indexing),
           retrieval/dense_pgvector_retriever.py (query encoding)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from . import config


def load_sentence_transformer(model_name: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

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
