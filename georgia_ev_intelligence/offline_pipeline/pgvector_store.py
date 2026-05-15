"""Store child chunks as vectors in Neon PostgreSQL using pgvector."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from georgia_ev_intelligence.shared import config
from georgia_ev_intelligence.shared.embeddings import as_document_text, load_sentence_transformer

if TYPE_CHECKING:
    import psycopg2

    from .chunking.operations import ChunkingArtifacts


@dataclass(frozen=True)
class PgVectorIndexStats:
    chunks_indexed: int
    vector_size: int
    embedding_model: str


_CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS child_chunks (
    chunk_id           TEXT PRIMARY KEY,
    parent_record_id   TEXT NOT NULL,
    chunk_type         TEXT NOT NULL,
    source_type        TEXT NOT NULL,
    source_row_id      INTEGER NOT NULL,
    metadata           JSONB,
    embedding          VECTOR({vector_size}),
    created_at         TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS child_chunks_embedding_idx
ON child_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
"""

_UPSERT_SQL = """
INSERT INTO child_chunks (
    chunk_id, parent_record_id, chunk_type, source_type,
    source_row_id, metadata, embedding
) VALUES %s
ON CONFLICT (chunk_id) DO UPDATE SET
    parent_record_id = EXCLUDED.parent_record_id,
    chunk_type       = EXCLUDED.chunk_type,
    source_type      = EXCLUDED.source_type,
    source_row_id    = EXCLUDED.source_row_id,
    metadata         = EXCLUDED.metadata,
    embedding        = EXCLUDED.embedding;
"""


def _get_connection() -> "psycopg2.extensions.connection":
    import psycopg2

    url = config.NEON_DATABASE_URL
    if not url:
        raise RuntimeError(
            "NEON_DATABASE_URL is not set. Add it to your .env file."
        )
    return psycopg2.connect(url)


def _create_child_chunks_table(conn: "psycopg2.extensions.connection", vector_size: int) -> None:
    with conn.cursor() as cur:
        cur.execute(_CREATE_EXTENSION_SQL)
        cur.execute(_CREATE_TABLE_SQL.format(vector_size=vector_size))
        cur.execute(_CREATE_INDEX_SQL)


def _upsert_child_chunks(
    rows: list[tuple],
    conn: "psycopg2.extensions.connection",
) -> None:
    import psycopg2.extras

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, _UPSERT_SQL, rows, template=None, page_size=100)


def index_kb_children(
    artifacts: ChunkingArtifacts,
    model_name: str | None = None,
    batch_size: int | None = None,
) -> PgVectorIndexStats:
    """Encode child chunk embedding texts and upsert into the child_chunks pgvector table."""
    model_id = model_name or config.EMBEDDING_MODEL
    size = batch_size or config.PGVECTOR_BATCH_SIZE

    model = load_sentence_transformer(model_id)
    if hasattr(model, "get_embedding_dimension"):
        vector_size = int(model.get_embedding_dimension())
    else:
        vector_size = int(model.get_sentence_embedding_dimension())

    # Build parent_record_id → source_row_id lookup
    parent_row_id_map = {p.record_id: p.source_row_id for p in artifacts.parents}

    conn = _get_connection()
    try:
        _create_child_chunks_table(conn, vector_size)

        total = 0
        for batch in _batched(artifacts.children, size):
            texts = [as_document_text(c.embedding_text) for c in batch]
            vectors = model.encode(
                texts,
                batch_size=size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            rows = [
                (
                    chunk.chunk_id,
                    chunk.parent_record_id,
                    chunk.chunk_type.value,
                    chunk.source_type,
                    parent_row_id_map.get(chunk.parent_record_id, 0),
                    json.dumps(chunk.metadata),
                    vectors[i].astype(float).tolist(),
                )
                for i, chunk in enumerate(batch)
            ]
            _upsert_child_chunks(rows, conn)
            total += len(batch)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return PgVectorIndexStats(
        chunks_indexed=total,
        vector_size=vector_size,
        embedding_model=model_id,
    )


def _batched(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start: start + size]
