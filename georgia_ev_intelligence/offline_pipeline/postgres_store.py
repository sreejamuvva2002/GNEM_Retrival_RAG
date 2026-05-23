"""Store parent chunks in Neon PostgreSQL."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from georgia_ev_intelligence.shared import config

if TYPE_CHECKING:
    import psycopg2

    from .chunking.parent_chunk import ParentRecord


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS parent_chunks (
    record_id                    TEXT PRIMARY KEY,
    source_row_id                INTEGER,
    source_type                  TEXT,
    company                      TEXT,
    category                     TEXT,
    industry_group               TEXT,
    updated_location             TEXT,
    address                      TEXT,
    latitude                     NUMERIC,
    longitude                    NUMERIC,
    primary_facility_type        TEXT,
    ev_supply_chain_role         TEXT,
    primary_oems                 TEXT,
    supplier_or_affiliation_type TEXT,
    employment                   NUMERIC,
    product_service              TEXT,
    ev_battery_relevant          TEXT,
    classification_method        TEXT,
    row_id                       INTEGER,
    raw_row                      JSONB,
    parent_chunk_text            TEXT,
    created_at                   TIMESTAMPTZ DEFAULT NOW(),
    updated_at                   TIMESTAMPTZ DEFAULT NOW()
);
"""

_UPSERT_SQL = """
INSERT INTO parent_chunks (
    record_id, source_row_id, source_type,
    company, category, industry_group, updated_location,
    address, latitude, longitude, primary_facility_type,
    ev_supply_chain_role, primary_oems, supplier_or_affiliation_type,
    employment, product_service, ev_battery_relevant,
    classification_method, row_id, raw_row, parent_chunk_text,
    updated_at
) VALUES %s
ON CONFLICT (record_id) DO UPDATE SET
    source_row_id                = EXCLUDED.source_row_id,
    source_type                  = EXCLUDED.source_type,
    company                      = EXCLUDED.company,
    category                     = EXCLUDED.category,
    industry_group               = EXCLUDED.industry_group,
    updated_location             = EXCLUDED.updated_location,
    address                      = EXCLUDED.address,
    latitude                     = EXCLUDED.latitude,
    longitude                    = EXCLUDED.longitude,
    primary_facility_type        = EXCLUDED.primary_facility_type,
    ev_supply_chain_role         = EXCLUDED.ev_supply_chain_role,
    primary_oems                 = EXCLUDED.primary_oems,
    supplier_or_affiliation_type = EXCLUDED.supplier_or_affiliation_type,
    employment                   = EXCLUDED.employment,
    product_service              = EXCLUDED.product_service,
    ev_battery_relevant          = EXCLUDED.ev_battery_relevant,
    classification_method        = EXCLUDED.classification_method,
    row_id                       = EXCLUDED.row_id,
    raw_row                      = EXCLUDED.raw_row,
    parent_chunk_text            = EXCLUDED.parent_chunk_text,
    updated_at                   = NOW();
"""


def _get_connection() -> "psycopg2.extensions.connection":
    import psycopg2

    url = config.NEON_DATABASE_URL
    if not url:
        raise RuntimeError(
            "NEON_DATABASE_URL is not set. Add it to your .env file."
        )
    return psycopg2.connect(url)


def _create_parent_chunks_table(conn: "psycopg2.extensions.connection") -> None:
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE_SQL)


def _upsert_parent_chunks(
    parents: list[ParentRecord],
    conn: "psycopg2.extensions.connection",
) -> int:
    import psycopg2.extras

    rows = [
        (
            p.record_id,
            p.source_row_id,
            p.source_type,
            p.company,
            p.category,
            p.industry_group,
            p.updated_location,
            p.address,
            _to_numeric(p.latitude),
            _to_numeric(p.longitude),
            p.primary_facility_type,
            p.ev_supply_chain_role,
            p.primary_oems,
            p.supplier_or_affiliation_type,
            _to_numeric(p.employment),
            p.product_service,
            p.ev_battery_relevant,
            p.classification_method,
            p.row_id,
            json.dumps(p.raw_row),
            p.parent_chunk_text,
            "NOW()",
        )
        for p in parents
    ]

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, _UPSERT_SQL, rows, template=None, page_size=100)

    return len(rows)


def store_parents_postgres(parents: list[ParentRecord]) -> int:
    """Create the parent_chunks table if needed and upsert all parent records.

    Returns the number of rows upserted.
    """
    conn = _get_connection()
    try:
        _create_parent_chunks_table(conn)
        count = _upsert_parent_chunks(parents, conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return count


def _to_numeric(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# raw_documents — web KB source document store
# ---------------------------------------------------------------------------

_CREATE_RAW_DOCS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS raw_documents (
    doc_id            TEXT PRIMARY KEY,          -- sha256 of normalised body_text
    url               TEXT NOT NULL,
    domain            TEXT,
    source_type       TEXT,                      -- company_site | gov_doc | news
    title             TEXT,
    body_text         TEXT,
    crawled_at        TIMESTAMPTZ,
    content_hash      TEXT,
    http_status       INT,
    language          TEXT DEFAULT 'en',
    linked_company_id TEXT,
    ingestion_status  TEXT DEFAULT 'new',        -- new | indexed | error
    error_detail      TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_raw_docs_status ON raw_documents(ingestion_status);
CREATE INDEX IF NOT EXISTS idx_raw_docs_domain  ON raw_documents(domain);
CREATE INDEX IF NOT EXISTS idx_raw_docs_url     ON raw_documents(url);
"""

_UPSERT_RAW_DOC_SQL = """
INSERT INTO raw_documents (
    doc_id, url, domain, source_type, title, body_text,
    crawled_at, content_hash, http_status, language,
    linked_company_id, ingestion_status, updated_at
) VALUES (
    %(doc_id)s, %(url)s, %(domain)s, %(source_type)s, %(title)s, %(body_text)s,
    %(crawled_at)s, %(content_hash)s, %(http_status)s, %(language)s,
    %(linked_company_id)s, %(ingestion_status)s, NOW()
)
ON CONFLICT (doc_id) DO UPDATE SET
    url               = EXCLUDED.url,
    domain            = EXCLUDED.domain,
    source_type       = EXCLUDED.source_type,
    title             = EXCLUDED.title,
    body_text         = EXCLUDED.body_text,
    crawled_at        = EXCLUDED.crawled_at,
    content_hash      = EXCLUDED.content_hash,
    http_status       = EXCLUDED.http_status,
    language          = EXCLUDED.language,
    linked_company_id = EXCLUDED.linked_company_id,
    ingestion_status  = EXCLUDED.ingestion_status,
    updated_at        = NOW();
"""

_FETCH_NEW_DOCS_SQL = """
SELECT doc_id, url, domain, source_type, title, body_text,
       crawled_at, content_hash, linked_company_id
FROM   raw_documents
WHERE  ingestion_status = 'new'
ORDER  BY crawled_at ASC
LIMIT  %(limit)s;
"""

_UPDATE_STATUS_SQL = """
UPDATE raw_documents
SET    ingestion_status = %(status)s,
       error_detail     = %(error_detail)s,
       updated_at       = NOW()
WHERE  doc_id = ANY(%(doc_ids)s::text[]);
"""


def ensure_raw_documents_table() -> None:
    """Create the raw_documents table and indexes if they don't exist."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_CREATE_RAW_DOCS_TABLE_SQL)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_raw_document(doc: dict) -> None:
    """Upsert a single RawDocument dict into raw_documents.

    ``doc`` must contain the keys that match the INSERT columns above.
    Missing optional keys default to None.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_UPSERT_RAW_DOC_SQL, {
                "doc_id":            doc.get("doc_id"),
                "url":               doc.get("url"),
                "domain":            doc.get("domain"),
                "source_type":       doc.get("source_type", "unknown"),
                "title":             doc.get("title"),
                "body_text":         doc.get("body_text"),
                "crawled_at":        doc.get("crawled_at"),
                "content_hash":      doc.get("content_hash"),
                "http_status":       doc.get("http_status"),
                "language":          doc.get("language", "en"),
                "linked_company_id": doc.get("linked_company_id"),
                "ingestion_status":  doc.get("ingestion_status", "new"),
            })
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_new_raw_documents(limit: int = 500) -> list[dict]:
    """Return up to *limit* raw_documents rows where ingestion_status = 'new'."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_FETCH_NEW_DOCS_SQL, {"limit": limit})
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def update_raw_doc_status(
    doc_ids: list[str],
    status: str,
    error_detail: str | None = None,
) -> None:
    """Bulk-update ingestion_status for a list of doc_ids."""
    if not doc_ids:
        return
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_UPDATE_STATUS_SQL, {
                "status":       status,
                "error_detail": error_detail,
                "doc_ids":      doc_ids,
            })
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
