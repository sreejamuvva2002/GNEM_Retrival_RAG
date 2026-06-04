"""Dense semantic retrieval over child chunks using pgvector cosine search."""
from __future__ import annotations

import json

import psycopg2

from ...shared import config
from ...shared.embeddings import as_query_text, load_sentence_transformer
from ..schemas import RetrievedChildChunk


_SEARCH_SQL = """
SELECT
    chunk_id,
    parent_record_id,
    chunk_type,
    metadata
FROM child_chunks
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""


class DensePgvectorRetriever:
    """Embed the user query and search child chunk embeddings via pgvector."""

    def __init__(self) -> None:
        self._model = load_sentence_transformer(config.EMBEDDING_MODEL)

    def search(self, query: str, top_k: int = 100) -> list[RetrievedChildChunk]:
        query_vec = self._model.encode(
            [as_query_text(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(float).tolist()

        conn = psycopg2.connect(config.NEON_DATABASE_URL)
        try:
            with conn.cursor() as cur:
                cur.execute(_SEARCH_SQL, (query_vec, top_k))
                rows = cur.fetchall()
        finally:
            conn.close()

        results: list[RetrievedChildChunk] = []
        for chunk_id, parent_record_id, chunk_type, metadata in rows:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append(RetrievedChildChunk(
                chunk_id=chunk_id,
                parent_record_id=parent_record_id,
                chunk_type=chunk_type,
                metadata=metadata or {},
            ))

        return results
