"""Safe column allowlist for structured queries against ``parent_chunks``.

The executor never interpolates an LLM/route-supplied identifier into SQL
unless it appears here. This is the single source of truth that keeps
``structured_sql`` / ``exact_lookup`` injection-safe: filter keys, requested
columns, group-by and sort-by fields are all checked against ``ALLOWED_COLUMNS``
before they reach a query string. Values are always passed as bound parameters.
"""
from __future__ import annotations

# Exactly the queryable text/numeric columns of the ``parent_chunks`` table
# (see offline_pipeline/postgres_store.py). Bookkeeping columns (record_id,
# raw_row, parent_chunk_text, timestamps, source ids) are intentionally omitted.
ALLOWED_COLUMNS: frozenset[str] = frozenset({
    "company",
    "category",
    "state",
    "industry_group",
    "updated_location",
    "address",
    "latitude",
    "longitude",
    "primary_facility_type",
    "ev_supply_chain_role",
    "primary_oems",
    "supplier_or_affiliation_type",
    "employment",
    "product_service",
    "ev_battery_relevant",
    "classification_method",
})

# Columns stored as NUMERIC — numeric comparison operators (GT/LT/BETWEEN …)
# may target these without an explicit cast surprise.
NUMERIC_COLUMNS: frozenset[str] = frozenset({
    "latitude",
    "longitude",
    "employment",
})

# Default column ordering when a route requests no specific columns.
DEFAULT_COLUMNS: tuple[str, ...] = (
    "company",
    "category",
    "ev_supply_chain_role",
    "product_service",
    "updated_location",
)


class UnknownColumnError(ValueError):
    """Raised when a route references a column outside the allowlist."""


def ensure_allowed(column: str) -> str:
    """Return ``column`` if it is allowlisted, else raise ``UnknownColumnError``."""
    if column not in ALLOWED_COLUMNS:
        raise UnknownColumnError(
            f"Column {column!r} is not in the safe allowlist: "
            f"{sorted(ALLOWED_COLUMNS)}"
        )
    return column
