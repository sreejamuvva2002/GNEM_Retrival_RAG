"""BM25 sparse retrieval over child chunks loaded from PostgreSQL."""
from __future__ import annotations

import json
import re
import threading
from typing import Any

import numpy as np
import psycopg2
from rank_bm25 import BM25Okapi

from ...shared import config
from ..schemas import RetrievedChildChunk


_LOAD_CHUNKS_SQL = """
SELECT chunk_id, parent_record_id, chunk_type, metadata
FROM child_chunks;
"""


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize text for BM25 with awareness of domain-specific patterns.

    Handles:
    - Hyphenated compounds (Hyundai-Kia) -> [hyundai-kia, hyundai, kia]
    - Slash-numeric patterns (Tier 1/2) -> [1/2, 1, 2]
    - Simple possessives (company's) -> [company]
    - Trailing punctuation stripped
    """
    text = text.lower()
    # Remove possessives
    text = re.sub(r"'s\b", "", text)
    # Find word tokens including hyphenated/slash compounds
    raw_tokens = re.findall(r"[\w]+(?:[/\-][\w]+)*", text)
    tokens: list[str] = []
    for t in raw_tokens:
        # Strip trailing punctuation artifacts
        t = t.strip(".,;:!?()[]{}\"'")
        if not t:
            continue
        tokens.append(t)
        # Expand hyphenated and slash compounds into sub-tokens
        if "-" in t or "/" in t:
            parts = re.split(r"[-/]", t)
            tokens.extend(p for p in parts if p)
    return tokens


def _build_bm25_text(chunk_type: str, metadata: dict[str, Any]) -> str:
    """Build structured BM25 text from child chunk metadata, preserving field names."""
    parts = [f"chunk_type: {chunk_type}"]
    for field_name, value in metadata.items():
        str_value = str(value).strip() if value is not None else ""
        if not str_value or str_value.lower() == "unknown":
            continue
        parts.append(f"{field_name}: {str_value}")
    return " ".join(parts)


class BM25Retriever:
    """BM25 sparse retrieval over child chunks from PostgreSQL."""

    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._loaded = False
        self._lock = threading.Lock()

    def _load(self) -> None:
        """Load child chunks from PostgreSQL and build BM25 index.

        Thread-safe: concurrent first requests will not double-load.
        """
        if self._loaded:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._loaded:
                return

            conn = psycopg2.connect(config.NEON_DATABASE_URL)
            try:
                with conn.cursor() as cur:
                    cur.execute(_LOAD_CHUNKS_SQL)
                    rows = cur.fetchall()
            finally:
                conn.close()

            corpus_tokens: list[list[str]] = []
            for chunk_id, parent_record_id, chunk_type, metadata in rows:
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                metadata = metadata or {}

                self._chunks.append({
                    "chunk_id": chunk_id,
                    "parent_record_id": parent_record_id,
                    "chunk_type": chunk_type,
                    "metadata": metadata,
                })

                text = _build_bm25_text(chunk_type, metadata)
                corpus_tokens.append(tokenize_bm25(text))

            self._bm25 = BM25Okapi(corpus_tokens)
            self._loaded = True

    def search(self, query: str, top_k: int = 100) -> list[RetrievedChildChunk]:
        self._load()
        if self._bm25 is None or not self._chunks:
            return []

        query_tokens = tokenize_bm25(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        n = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[-n:][::-1]
        top_indices = [int(i) for i in top_indices if scores[i] > 0]

        results: list[RetrievedChildChunk] = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            results.append(RetrievedChildChunk(
                chunk_id=chunk["chunk_id"],
                parent_record_id=chunk["parent_record_id"],
                chunk_type=chunk["chunk_type"],
                metadata=chunk["metadata"],
            ))

        return results
