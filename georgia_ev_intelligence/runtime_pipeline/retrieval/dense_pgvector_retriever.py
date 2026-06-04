"""Dense semantic retrieval over child chunks using pgvector cosine search.

WHY THIS FILE EXISTS
--------------------
Provides vector/semantic search over child chunks stored in Neon PostgreSQL with
the pgvector extension.  Dense retrieval captures conceptual similarity and
paraphrase matches that BM25 (keyword) search may miss.  Together, BM25 + dense
form the hybrid retrieval backbone.

TECHNIQUE: Dense Retrieval with pgvector
-----------------------------------------
- Encodes the query using the ``nomic-ai/nomic-embed-text-v1.5`` sentence
  transformer (768 dimensions).
- The query embedding is prefixed with ``"search_query:"`` (asymmetric embedding
  convention) to match document embeddings indexed with ``"search_document:"``.
- Runs a single SQL query using the pgvector ``<=>`` cosine distance operator
  to find the closest child chunks in embedding space.
- Returns the top-K results in order of cosine similarity (ascending distance).

ASYMMETRIC EMBEDDINGS
---------------------
``as_query_text(query)`` from ``shared.embeddings`` prepends ``"search_query:"``.
At indexing time, ``as_document_text(text)`` prepends ``"search_document:"``
(see ``offline_pipeline/``).  This asymmetry is the recommended usage for
Nomic Embed models and significantly improves retrieval precision.

CORRECTNESS CONTRACT
--------------------
- Returns ``RetrievedChildChunk`` objects (metadata only, no text embeddings).
- A new DB connection is opened and closed per query call (connection pooling is
  handled at the Neon/cloud layer for now).
- Results feed into the same ``ChildResultMerger`` → ``ParentChildMapper`` →
  ``CrossEncoderReranker`` pipeline as BM25 results.
"""
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
