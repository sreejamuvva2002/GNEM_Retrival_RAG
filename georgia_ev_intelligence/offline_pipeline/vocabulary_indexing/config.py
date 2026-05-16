"""Vocabulary indexing configuration and constants."""
from __future__ import annotations

TABLE_NAME = "kb_vocabulary_terms"

EXCEL_FILENAME = "vocabulary_index_export.xlsx"

# Values to skip during term extraction
SKIP_VALUES: frozenset[str] = frozenset(
    {"", "unknown", "n/a", "none", "nan", "null", "na"}
)

# Mapping: normalized DataFrame column name -> semantic term_type
COLUMN_TERM_TYPE_MAP: dict[str, str] = {
    "company": "company",
    "category": "supplier_tier",
    "industry_group": "industry_group",
    "updated_location": "location",
    "address": "address",
    "primary_facility_type": "facility_type",
    "ev_supply_chain_role": "ev_supply_chain_role",
    "primary_oems": "primary_oem",
    "supplier_or_affiliation_type": "supplier_affiliation_type",
    "product_service": "product_service",
    "ev_battery_relevant": "ev_relevance",
    "classification_method": "classification_method",
    "employment": "numeric_attribute",
    "latitude": "geo_coordinate",
    "longitude": "geo_coordinate",
}

# Columns whose cell values may contain multiple terms separated by delimiters
MULTI_VALUE_COLUMNS: frozenset[str] = frozenset({"primary_oems", "product_service"})

# Numeric columns - indexed as controlled vocabulary, not individual values
NUMERIC_COLUMNS: frozenset[str] = frozenset({"employment", "latitude", "longitude"})

# Separator pattern for splitting multi-value columns.
# Does NOT include "/" to preserve values like "tier 1/2", "tier 2/3".
SEPARATOR_PATTERN = r"[,;|\n]"
