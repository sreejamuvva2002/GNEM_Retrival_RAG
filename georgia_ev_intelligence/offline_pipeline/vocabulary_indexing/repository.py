"""PostgreSQL operations for the vocabulary index."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from georgia_ev_intelligence.shared import config

from .config import TABLE_NAME
from .models import VocabularyTerm

if TYPE_CHECKING:
    import psycopg2.extensions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------

_CREATE_EXTENSION_VECTOR = "CREATE EXTENSION IF NOT EXISTS vector;"
_CREATE_EXTENSION_TRGM = "CREATE EXTENSION IF NOT EXISTS pg_trgm;"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    id SERIAL PRIMARY KEY,
    normalized_value TEXT NOT NULL,
    term_frequency INTEGER NOT NULL,
    row_ids INTEGER[] NOT NULL,
    multiple_words BOOLEAN NOT NULL,
    term_type TEXT NOT NULL,
    source_column TEXT NOT NULL,
    term_vector VECTOR({vector_size}),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_INDEXES_SQL = [
    f"CREATE INDEX IF NOT EXISTS idx_vocab_normalized_value ON {TABLE_NAME} (normalized_value);",
    f"CREATE INDEX IF NOT EXISTS idx_vocab_term_type ON {TABLE_NAME} (term_type);",
    f"CREATE INDEX IF NOT EXISTS idx_vocab_source_column ON {TABLE_NAME} (source_column);",
]

_CREATE_TRGM_INDEX_SQL = (
    f"CREATE INDEX IF NOT EXISTS idx_vocab_trgm ON {TABLE_NAME} "
    "USING gin (normalized_value gin_trgm_ops);"
)

_CREATE_VECTOR_INDEX_SQL = (
    f"CREATE INDEX IF NOT EXISTS idx_vocab_vector ON {TABLE_NAME} "
    "USING ivfflat (term_vector vector_cosine_ops) WITH (lists = 50);"
)

_TRUNCATE_SQL = f"TRUNCATE TABLE {TABLE_NAME};"

_INSERT_SQL = f"""
INSERT INTO {TABLE_NAME} (
    normalized_value, term_frequency, row_ids, multiple_words,
    term_type, source_column, term_vector
) VALUES %s;
"""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def get_connection() -> "psycopg2.extensions.connection":
    """Create a new PostgreSQL connection using project config."""
    import psycopg2

    url = config.NEON_DATABASE_URL
    if not url:
        raise RuntimeError("NEON_DATABASE_URL is not set. Add it to your .env file.")
    return psycopg2.connect(url)


def ensure_table(
    conn: "psycopg2.extensions.connection", vector_size: int
) -> None:
    """Create the vocabulary table and indexes.

    Handles missing extensions gracefully using SAVEPOINTs.
    Logs warnings and skips optional indexes if extensions are unavailable.
    """
    with conn.cursor() as cur:
        # Try to enable pgvector (optional - needed for VECTOR type and index)
        _try_execute(
            cur, _CREATE_EXTENSION_VECTOR,
            "pgvector extension enabled.",
            "Could not enable pgvector extension (VECTOR column will be TEXT fallback): %s",
        )

        # Try to enable pg_trgm (optional - needed for trigram index)
        _try_execute(
            cur, _CREATE_EXTENSION_TRGM,
            "pg_trgm extension enabled.",
            "Could not enable pg_trgm extension: %s",
        )

        # Create table
        cur.execute(
            _CREATE_TABLE_SQL.format(table=TABLE_NAME, vector_size=vector_size)
        )

        # Standard B-tree indexes (always succeed)
        for sql in _CREATE_INDEXES_SQL:
            cur.execute(sql)

        # Trigram index (optional - requires pg_trgm)
        _try_execute(
            cur, _CREATE_TRGM_INDEX_SQL,
            "Trigram index created.",
            "Could not create trigram index (pg_trgm may be unavailable): %s",
        )

        # Vector index (optional - requires pgvector)
        _try_execute(
            cur, _CREATE_VECTOR_INDEX_SQL,
            "Vector index created.",
            "Could not create vector index (pgvector may be unavailable): %s",
        )

    conn.commit()


def _try_execute(
    cur: "psycopg2.extensions.cursor",
    sql: str,
    success_msg: str,
    failure_msg: str,
) -> bool:
    """Execute SQL within a SAVEPOINT. Roll back to savepoint on failure."""
    try:
        cur.execute("SAVEPOINT optional_op")
        cur.execute(sql)
        cur.execute("RELEASE SAVEPOINT optional_op")
        logger.info(success_msg)
        return True
    except Exception as exc:
        cur.execute("ROLLBACK TO SAVEPOINT optional_op")
        logger.warning(failure_msg, exc)
        return False


def truncate_table(conn: "psycopg2.extensions.connection") -> None:
    """Truncate the vocabulary table for idempotent rebuild."""
    with conn.cursor() as cur:
        cur.execute(_TRUNCATE_SQL)
    conn.commit()
    logger.info("Truncated table %s for rebuild.", TABLE_NAME)


def insert_batch(
    conn: "psycopg2.extensions.connection",
    terms: list[VocabularyTerm],
    page_size: int = 100,
) -> int:
    """Batch insert vocabulary terms using execute_values.

    Returns the number of terms inserted.
    """
    import psycopg2.extras

    rows = [
        (
            term.normalized_value,
            term.term_frequency,
            term.row_ids,
            term.multiple_words,
            term.term_type,
            term.source_column,
            term.term_vector.astype(float).tolist() if term.term_vector is not None else None,
        )
        for term in terms
    ]

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur, _INSERT_SQL, rows, template=None, page_size=page_size
        )

    return len(rows)
