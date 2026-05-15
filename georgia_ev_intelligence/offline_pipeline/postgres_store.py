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
