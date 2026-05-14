"""PostgreSQL storage for complete parent chunks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from georgia_ev_intelligence.indexing.chunking import ParentRecord

try:
    import psycopg
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    psycopg = None
    Jsonb = None


CREATE_PARENT_CHUNKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS parent_chunks (
    record_id TEXT PRIMARY KEY,
    source_row_id TEXT,
    source_type TEXT NOT NULL DEFAULT 'excel',

    company TEXT,
    category TEXT,
    industry_group TEXT,
    updated_location TEXT,
    address TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    primary_facility_type TEXT,
    ev_supply_chain_role TEXT,
    primary_oems TEXT,
    supplier_or_affiliation_type TEXT,
    employment INTEGER,
    product_service TEXT,
    ev_battery_relevant TEXT,
    classification_method TEXT,
    row_id TEXT,

    raw_row JSONB NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""


CREATE_PARENT_CHUNKS_INDEXES_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_company "
    "ON parent_chunks(company);",
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_category "
    "ON parent_chunks(category);",
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_industry_group "
    "ON parent_chunks(industry_group);",
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_updated_location "
    "ON parent_chunks(updated_location);",
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_ev_role "
    "ON parent_chunks(ev_supply_chain_role);",
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_primary_oems "
    "ON parent_chunks(primary_oems);",
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_supplier_type "
    "ON parent_chunks(supplier_or_affiliation_type);",
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_employment "
    "ON parent_chunks(employment);",
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_ev_relevant "
    "ON parent_chunks(ev_battery_relevant);",
    "CREATE INDEX IF NOT EXISTS idx_parent_chunks_raw_row_gin "
    "ON parent_chunks USING GIN(raw_row);",
)


UPSERT_PARENT_CHUNK_SQL = """
INSERT INTO parent_chunks (
    record_id,
    source_row_id,
    source_type,
    company,
    category,
    industry_group,
    updated_location,
    address,
    latitude,
    longitude,
    primary_facility_type,
    ev_supply_chain_role,
    primary_oems,
    supplier_or_affiliation_type,
    employment,
    product_service,
    ev_battery_relevant,
    classification_method,
    row_id,
    raw_row
) VALUES (
    %(record_id)s,
    %(source_row_id)s,
    %(source_type)s,
    %(company)s,
    %(category)s,
    %(industry_group)s,
    %(updated_location)s,
    %(address)s,
    %(latitude)s,
    %(longitude)s,
    %(primary_facility_type)s,
    %(ev_supply_chain_role)s,
    %(primary_oems)s,
    %(supplier_or_affiliation_type)s,
    %(employment)s,
    %(product_service)s,
    %(ev_battery_relevant)s,
    %(classification_method)s,
    %(row_id)s,
    %(raw_row)s
)
ON CONFLICT (record_id) DO UPDATE SET
    source_row_id = EXCLUDED.source_row_id,
    source_type = EXCLUDED.source_type,
    company = EXCLUDED.company,
    category = EXCLUDED.category,
    industry_group = EXCLUDED.industry_group,
    updated_location = EXCLUDED.updated_location,
    address = EXCLUDED.address,
    latitude = EXCLUDED.latitude,
    longitude = EXCLUDED.longitude,
    primary_facility_type = EXCLUDED.primary_facility_type,
    ev_supply_chain_role = EXCLUDED.ev_supply_chain_role,
    primary_oems = EXCLUDED.primary_oems,
    supplier_or_affiliation_type = EXCLUDED.supplier_or_affiliation_type,
    employment = EXCLUDED.employment,
    product_service = EXCLUDED.product_service,
    ev_battery_relevant = EXCLUDED.ev_battery_relevant,
    classification_method = EXCLUDED.classification_method,
    row_id = EXCLUDED.row_id,
    raw_row = EXCLUDED.raw_row,
    updated_at = NOW();
"""


@dataclass(frozen=True)
class ParentChunkPostgresStats:
    upserted_count: int
    table_count: int


class ParentChunkPostgresStore:
    """Store complete parent chunks in PostgreSQL."""

    def __init__(self, database_url: str | None = None, **connect_kwargs: Any):
        if not database_url and not connect_kwargs:
            raise ValueError("A PostgreSQL database URL or connection parameters are required.")

        if psycopg is None:
            raise RuntimeError(
                "psycopg is required for PostgreSQL parent chunk storage. "
                "Install dependencies from requirements.txt."
            )

        self._conn = psycopg.connect(database_url or "", **connect_kwargs)

    def create_schema(self) -> None:
        """Create the parent_chunks table and indexes if they do not exist."""
        with self._conn.cursor() as cur:
            cur.execute(CREATE_PARENT_CHUNKS_TABLE_SQL)
            for sql in CREATE_PARENT_CHUNKS_INDEXES_SQL:
                cur.execute(sql)
        self._conn.commit()

    def upsert_parent_chunks(self, parents: list[ParentRecord]) -> int:
        """
        Insert or update parent chunks by record_id.

        Existing rows keep created_at and receive updated_at = NOW().
        """
        if not parents:
            return 0

        records = [_parent_to_sql_record(parent) for parent in parents]

        with self._conn.cursor() as cur:
            cur.executemany(UPSERT_PARENT_CHUNK_SQL, records)
        self._conn.commit()

        return len(records)

    def count_parent_chunks(self) -> int:
        """Return total rows currently stored in parent_chunks."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM parent_chunks;")
            row = cur.fetchone()

        return int(row[0]) if row else 0

    def store_parent_chunks(self, parents: list[ParentRecord]) -> ParentChunkPostgresStats:
        """Create schema, upsert parent chunks, and return table count."""
        self.create_schema()
        upserted = self.upsert_parent_chunks(parents)
        table_count = self.count_parent_chunks()
        return ParentChunkPostgresStats(
            upserted_count=upserted,
            table_count=table_count,
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ParentChunkPostgresStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _parent_to_sql_record(parent: ParentRecord) -> dict[str, Any]:
    raw_row = parent.payload()
    row_data = parent.row_data

    return {
        "record_id": parent.record_id,
        "source_row_id": _to_text(parent.source_row_id),
        "source_type": "excel",
        "company": _to_text(row_data.get("company")),
        "category": _to_text(row_data.get("category")),
        "industry_group": _to_text(row_data.get("industry_group")),
        "updated_location": _to_text(row_data.get("updated_location")),
        "address": _to_text(row_data.get("address")),
        "latitude": _to_float(row_data.get("latitude")),
        "longitude": _to_float(row_data.get("longitude")),
        "primary_facility_type": _to_text(row_data.get("primary_facility_type")),
        "ev_supply_chain_role": _to_text(row_data.get("ev_supply_chain_role")),
        "primary_oems": _to_text(row_data.get("primary_oems")),
        "supplier_or_affiliation_type": _to_text(
            row_data.get("supplier_or_affiliation_type")
        ),
        "employment": _to_int(row_data.get("employment")),
        "product_service": _to_text(row_data.get("product_service")),
        "ev_battery_relevant": _to_text(row_data.get("ev_battery_relevant")),
        "classification_method": _to_text(row_data.get("classification_method")),
        "row_id": _to_text(row_data.get("_row_id")),
        "raw_row": _to_jsonb(raw_row),
    }


def _to_text(value: Any) -> str | None:
    if _is_missing(value):
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    return str(value)


def _to_float(value: Any) -> float | None:
    if _is_missing(value):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(number) or math.isinf(number):
        return None

    return number


def _to_int(value: Any) -> int | None:
    if _is_missing(value):
        return None

    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None

    if math.isnan(number) or math.isinf(number):
        return None

    return int(number)


def _to_jsonb(value: dict[str, Any]) -> Any:
    if Jsonb is None:
        return value

    return Jsonb(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    try:
        result = pd.isna(value)
    except Exception:
        return False

    if isinstance(result, bool):
        return result

    return False
