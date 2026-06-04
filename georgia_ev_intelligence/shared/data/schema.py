"""Column metadata index for the normalised Knowledge Base DataFrame.

WHY THIS FILE EXISTS:
  Builds a per-column index (ColumnMeta) that describes how each KB column
  should be interpreted for filtering and search.  It is used by query-building
  code that needs to know whether a column supports exact matching (e.g. Category
  has ≤ 60 distinct values) or partial/substring matching (e.g. Product/Service
  has long free-text entries).

TECHNIQUES:
  Heuristic-based classification:
    - Columns with ≤ 60 unique values and avg token length ≤ 50  → "exact" match
    - Columns with many unique values or long text               → "partial" match
    - primary_oems is always "partial" (PARTIAL_OVERRIDE_COLUMNS)
      because values are compound strings like "Hyundai Kia Rivian"
  Component extraction: for location columns that contain "City, County"
    style values, individual components are extracted so "Savannah" matches
    "Savannah, Chatham County".

DATA MODEL (ColumnMeta):
  unique_values   — sorted list of all distinct values in the column
  match_type      — "exact", "partial", or "numeric"
  is_numeric      — True for employment / lat / long
  is_filterable   — False for internal metadata columns
  components      — sub-parts of compound location strings

RELATIONSHIPS:
  Called by: query-building logic that needs to know how to filter KB rows
  Reads from: the normalised DataFrame produced by shared/data/loader.py
"""
from dataclasses import dataclass, field
import pandas as pd


# Internal metadata columns that should not be used as query filters.
# These are schema-level constants (column names), not data values.
NON_FILTER_COLUMNS = {"classification_method", "supplier_or_affiliation_type"}

# Columns to skip entirely (not searchable)
SKIP_COLUMNS = {"_row_id", "latitude", "longitude", "address"}

# primary_oems values are compound (e.g. "Hyundai Kia Rivian") → always partial-match
# so "Rivian" in a question matches both "Rivian Automotive" and "Hyundai Kia Rivian" rows.
PARTIAL_OVERRIDE_COLUMNS = {"primary_oems"}


@dataclass
class ColumnMeta:
    unique_values: list[str]
    match_type: str          # "exact", "partial", or "numeric"
    is_numeric: bool
    is_filterable: bool      # False for internal metadata columns
    components: list[str] = field(default_factory=list)


def build(df: pd.DataFrame) -> dict[str, ColumnMeta]:
    index: dict[str, ColumnMeta] = {}

    for col in df.columns:
        if col in SKIP_COLUMNS:
            continue

        series = df[col].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

        if is_numeric:
            index[col] = ColumnMeta(
                unique_values=[], match_type="numeric",
                is_numeric=True, is_filterable=False
            )
            continue

        values = sorted({str(v).strip() for v in series if str(v).strip() and str(v) != "nan"})

        # Heuristic: discrete categories → exact matching; free text → partial matching.
        # Threshold raised to 60 unique values to correctly classify ev_supply_chain_role.
        avg_len = sum(len(v) for v in values) / max(len(values), 1)
        if col in PARTIAL_OVERRIDE_COLUMNS:
            match_type = "partial"
        else:
            match_type = "exact" if len(values) <= 60 and avg_len <= 50 else "partial"

        # Extract sub-components for comma-separated location values ("City, County")
        components: list[str] = []
        if any("," in v for v in values[:5]):
            for v in values:
                parts = [p.strip() for p in v.split(",")]
                components.extend(p for p in parts if len(p) >= 3)
            components = sorted(set(components))

        index[col] = ColumnMeta(
            unique_values=values,
            match_type=match_type,
            is_numeric=False,
            is_filterable=(col not in NON_FILTER_COLUMNS),
            components=components,
        )

    return index