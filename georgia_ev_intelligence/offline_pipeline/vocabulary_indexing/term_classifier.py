"""Map source columns to semantic term types."""
from __future__ import annotations

from .config import COLUMN_TERM_TYPE_MAP


def classify(source_column: str) -> str:
    """Return the term_type for a given normalized DataFrame column name.

    Falls back to 'other' for unmapped columns.
    """
    return COLUMN_TERM_TYPE_MAP.get(source_column, "other")
