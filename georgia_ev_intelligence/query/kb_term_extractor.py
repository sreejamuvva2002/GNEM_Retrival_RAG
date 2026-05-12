"""
Dynamic KB vocabulary discovery from a retrieved candidate row set.

No data values are hardcoded.  All terms are mined from the DataFrame
passed in — which is itself built from live retrieval results.

Column weights reflect retrieval relevance; they are defined in terms of
match_type rather than hard column names wherever possible.  The two
sets _HIGH_WEIGHT_COLS and _MED_WEIGHT_COLS list column names that are
known from the live schema and therefore acceptable as metadata constants.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from ..data.schema import ColumnMeta, SKIP_COLUMNS

# Columns whose values are most discriminating for retrieval quality
_HIGH_WEIGHT_COLS: frozenset[str] = frozenset({
    "product_service",
    "ev_supply_chain_role",
    "supplier_type",
    "primary_oems",
})
_MED_WEIGHT_COLS: frozenset[str] = frozenset({"notes"})

# These columns carry entity identity, not retrieval vocabulary
_SKIP_EXTRACTION: frozenset[str] = SKIP_COLUMNS | {"company", "location", "_score"}

_MIN_TERM_CHARS = 4      # ignore single-letter or trivially short tokens
_MAX_NGRAM      = 3      # maximum phrase length extracted from free-text cells


# ── Tokenisation ──────────────────────────────────────────────────────────────

def _clean_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9\-]*", text.lower())


def _extract_phrases(cell_value: str) -> list[str]:
    """
    Yield 1–3-word token n-grams from a cell value, plus the whole
    stripped value (preserves longer multi-word phrases verbatim).
    """
    tokens = _clean_tokens(cell_value)
    phrases: list[str] = []
    for n in range(1, _MAX_NGRAM + 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if len(phrase) >= _MIN_TERM_CHARS:
                phrases.append(phrase)
    whole = cell_value.strip().lower()
    if len(whole) >= _MIN_TERM_CHARS and whole not in phrases:
        phrases.append(whole)
    return phrases


# ── Public API ────────────────────────────────────────────────────────────────

def extract(
    candidates_df: pd.DataFrame,
    schema_index: dict[str, ColumnMeta],
    probes: list[str],
    min_frequency: int = 2,
    top_n: int = 30,
) -> dict:
    """
    Discover KB vocabulary terms from a candidate row set.

    Parameters
    ----------
    candidates_df : DataFrame of retrieved KB rows (may include _score col)
    schema_index  : live schema metadata (column types, filterable flag)
    probes        : Stage-1 semantic probes (stored in term_sources for tracing)
    min_frequency : discard terms that appear in fewer rows than this
    top_n         : maximum distinct terms to return

    Returns
    -------
    {
        "kb_discovered_terms": list[str],
        "term_sources": list[{term, source_columns, supporting_row_ids,
                               frequency, weight}]
    }
    """
    if candidates_df.empty:
        return {"kb_discovered_terms": [], "term_sources": []}

    # term → aggregated statistics
    term_data: dict[str, dict] = defaultdict(lambda: {
        "source_columns": set(),
        "supporting_row_ids": [],
        "frequency": 0,
        "weight": 0.0,
    })

    row_id_series = (
        candidates_df["_row_id"]
        if "_row_id" in candidates_df.columns
        else candidates_df.index.astype(str)
    )

    for rel_idx, (abs_idx, row) in enumerate(candidates_df.iterrows()):
        row_id = str(
            row_id_series.iloc[rel_idx]
            if hasattr(row_id_series, "iloc")
            else rel_idx
        )

        for col, meta in schema_index.items():
            if col in _SKIP_EXTRACTION or col not in candidates_df.columns:
                continue
            if meta.is_numeric:
                continue

            cell = row.get(col)
            if cell is None or pd.isna(cell) or not str(cell).strip():
                continue

            weight = (
                2.0 if col in _HIGH_WEIGHT_COLS
                else 1.0 if col in _MED_WEIGHT_COLS
                else 0.5
            )
            cell_str = str(cell)

            if meta.match_type == "exact":
                # Categorical column — use the exact cell value
                term = cell_str.strip().lower()
                if len(term) >= _MIN_TERM_CHARS:
                    _record(term_data, term, col, row_id, weight)
            else:
                # Free-text column — extract n-gram phrases
                for phrase in _extract_phrases(cell_str):
                    _record(term_data, phrase, col, row_id, weight)

    # Filter by minimum row frequency, then rank by weight × frequency
    qualified = [
        (term, data)
        for term, data in term_data.items()
        if data["frequency"] >= min_frequency
    ]
    qualified.sort(
        key=lambda x: x[1]["weight"] * x[1]["frequency"],
        reverse=True,
    )
    top = qualified[:top_n]

    kb_discovered_terms = [t for t, _ in top]
    term_sources = [
        {
            "term": t,
            "source_columns": sorted(d["source_columns"]),
            "supporting_row_ids": list(dict.fromkeys(d["supporting_row_ids"]))[:5],
            "frequency": d["frequency"],
            "weight": round(d["weight"], 2),
        }
        for t, d in top
    ]

    return {"kb_discovered_terms": kb_discovered_terms, "term_sources": term_sources}


def _record(
    store: dict,
    term: str,
    col: str,
    row_id: str,
    weight: float,
) -> None:
    d = store[term]
    d["source_columns"].add(col)
    d["supporting_row_ids"].append(row_id)
    d["frequency"] += 1
    d["weight"] = max(d["weight"], weight)
