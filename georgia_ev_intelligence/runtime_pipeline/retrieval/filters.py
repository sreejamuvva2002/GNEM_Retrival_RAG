"""
DataFrame filter / mask-building utilities.

Extracted from reasoning/retriever.py to eliminate the cross-layer dependency
where retrieval/rag.py imported private functions from reasoning/retriever.py.

Both reasoning/retriever.py and retrieval/rag.py now import from this module.
"""
from __future__ import annotations

import re

import pandas as pd

from ...shared.data.schema import ColumnMeta


# Generic business suffixes that should not be used alone for OEM partial matching.
GENERIC_NAME_WORDS = {
    "corp",
    "inc",
    "llc",
    "ltd",
    "co",
    "group",
    "automotive",
    "manufacturing",
    "industries",
    "international",
    "america",
    "americas",
    "holdings",
    "enterprise",
    "enterprises",
    "company",
    "systems",
}

# Relationship-intent keywords -> primary_oems is the right filter column.
RELATIONSHIP_KEYWORDS = {
    "linked to",
    "supplier of",
    "supply chain",
    "supplier network",
    "connected to",
    "supplies to",
    "works with",
    "partners with",
}


def is_relationship_query(question: str) -> bool:
    q = (question or "").lower()
    return any(kw in q for kw in RELATIONSHIP_KEYWORDS)


def expand_partial_value(val: str) -> list[str]:
    """For a compound name like 'Rivian Automotive', extract significant words."""
    words = [w.strip(".,()") for w in str(val).split()]
    significant = [
        w for w in words
        if len(w) >= 4 and w.lower() not in GENERIC_NAME_WORDS
    ]
    return significant if significant else [str(val)]


# ── Mask builders ────────────────────────────────────────────────────────────

def build_col_mask(
    df: pd.DataFrame,
    col: str,
    values: list[str],
    schema_index: dict[str, ColumnMeta],
    allow_word_expansion: bool = False,
) -> pd.Series:
    """
    Build OR mask for one column.

    - Exact columns use equality first.
    - Partial columns use contains.
    - If exact equality finds no rows, safely tries contains as fallback because
      some KB categorical values may include suffixes such as "Supplier".
    """
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    meta = schema_index.get(col)
    series = df[col].astype(str)
    col_mask = pd.Series([False] * len(df), index=df.index)

    for val in values:
        val = str(val).strip()
        if not val:
            continue

        if meta and meta.match_type == "exact":
            exact_mask = series.str.lower() == val.lower()

            # Fallback for slash/category-style values: "Tier 1" should match
            # "Tier 1 Supplier"; exact values still remain safest first.
            if exact_mask.sum() == 0:
                contains_mask = series.str.contains(re.escape(val), case=False, na=False)
                col_mask = col_mask | contains_mask
            else:
                col_mask = col_mask | exact_mask
        else:
            full_mask = series.str.contains(re.escape(val), case=False, na=False)

            if allow_word_expansion and full_mask.sum() <= 2:
                expanded_mask = pd.Series([False] * len(df), index=df.index)
                for word in expand_partial_value(val):
                    expanded_mask = expanded_mask | series.str.contains(
                        re.escape(word),
                        case=False,
                        na=False,
                    )
                col_mask = col_mask | expanded_mask
            else:
                col_mask = col_mask | full_mask

    return col_mask


def build_and_mask(
    df: pd.DataFrame,
    filters: dict[str, list[str]],
    schema_index: dict[str, ColumnMeta],
) -> pd.Series:
    """AND across columns, OR within each column's values."""
    mask = pd.Series([True] * len(df), index=df.index)

    for col, values in (filters or {}).items():
        if col not in df.columns:
            continue
        mask = mask & build_col_mask(df, col, values, schema_index)

    return mask


def best_single_filter(
    df: pd.DataFrame,
    filters: dict[str, list[str]],
    schema_index: dict[str, ColumnMeta],
    question: str = "",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Return the best single-column filter result.

    For relationship queries ("linked to", "supplier of", etc.), prefer
    primary_oems over company so we return suppliers, not the OEM itself.
    """
    is_rel = is_relationship_query(question)

    candidates: list[tuple[int, int, str, pd.DataFrame, dict[str, list[str]]]] = []

    for col, values in (filters or {}).items():
        if col not in df.columns:
            continue

        if is_rel and col == "company" and "primary_oems" in filters:
            continue

        m = build_col_mask(
            df,
            col,
            values,
            schema_index,
            allow_word_expansion=True,
        )
        candidate = df[m]

        if len(candidate) > 0:
            meta = schema_index.get(col)
            exact_bonus = 0 if meta and meta.match_type == "exact" else 1
            candidates.append((exact_bonus, len(candidate), col, candidate, {col: values}))

    if not candidates:
        # Fall back without relationship preference.
        for col, values in (filters or {}).items():
            if col not in df.columns:
                continue

            m = build_col_mask(
                df,
                col,
                values,
                schema_index,
                allow_word_expansion=True,
            )
            candidate = df[m]

            if len(candidate) > 0:
                meta = schema_index.get(col)
                exact_bonus = 0 if meta and meta.match_type == "exact" else 1
                candidates.append((exact_bonus, len(candidate), col, candidate, {col: values}))

    if candidates:
        # Prefer exact-match column results; among those, pick most selective.
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, _, _, best_df, best_filter = candidates[0]
        return best_df, best_filter

    return df, {}
