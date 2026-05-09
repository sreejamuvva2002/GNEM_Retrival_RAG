"""
Applies matched filters to the KB DataFrame and handles aggregate / rank intents.
All operations are pure pandas — no external DB required.

Fallback cascade when strict AND produces 0 rows:
  1. Try strict AND of all filters
  2. If 0 rows: try each individual column filter, pick most selective (fewest rows)
  3. If still 0: return full DataFrame with a "low confidence" flag
"""
from __future__ import annotations
import re
import pandas as pd
from dataclasses import dataclass, field
from .term_matcher import MatchResult
from .schema_index import ColumnMeta


@dataclass
class RetrievalResult:
    rows: pd.DataFrame
    intent: dict
    total_matched: int
    filters_applied: dict[str, list[str]]
    support_level: str


# ── Intent detection ──────────────────────────────────────────────────────────

_RANK_HIGH  = {"highest", "most", "maximum", "largest", "top", "biggest", "greatest"}
_RANK_LOW   = {"lowest", "minimum", "fewest", "least", "smallest", "bottom"}
_SUM_WORDS  = {"total", "sum", "combined", "aggregate", "overall"}
_COUNT_WORDS = {"how many", "count", "number of", "total number"}
_SPOF_WORDS  = {
    "single point of failure", "sole supplier", "only one company",
    "only a single", "sole provider", "single supplier"
}


def _word_in(phrase: str, text: str) -> bool:
    """Whole-word match to avoid 'count' matching 'county'."""
    return bool(re.search(r"\b" + re.escape(phrase) + r"\b", text))


def _detect_intent(question: str) -> dict:
    q = question.lower()

    if any(phrase in q for phrase in _SPOF_WORDS):
        return {"type": "spof"}

    # Use word boundary matching so "count" doesn't trigger inside "county"
    for phrase in _COUNT_WORDS:
        if _word_in(phrase, q):
            return {"type": "count"}

    has_high = any(_word_in(w, q) for w in _RANK_HIGH)
    has_low  = any(_word_in(w, q) for w in _RANK_LOW)
    has_sum  = any(_word_in(w, q) for w in _SUM_WORDS)

    if has_sum and (_word_in("county", q) or _word_in("tier", q) or _word_in("role", q)):
        direction = "asc" if has_low else "desc"
        return {"type": "aggregate_sum", "direction": direction}

    if has_high or has_low:
        direction = "asc" if has_low else "desc"
        return {"type": "rank", "n": 1, "direction": direction}

    return {"type": "filter"}


# ── Column helpers ────────────────────────────────────────────────────────────

def _is_string_col(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _numeric_col(df: pd.DataFrame, question: str) -> str | None:
    q_words = set(re.split(r"\W+", question.lower()))
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        col_words = set(re.split(r"[_\W]+", col.lower()))
        if q_words & col_words:
            return col
    # Fallback: first numeric col that isn't an ID
    non_id = [c for c in numeric_cols if "id" not in c.lower() and c != "_row_id"]
    return non_id[0] if non_id else (numeric_cols[0] if numeric_cols else None)


def _group_col(df: pd.DataFrame, question: str) -> str | None:
    q_words = set(re.split(r"\W+", question.lower()))
    candidates = []

    for col in df.columns:
        if not _is_string_col(df[col]):
            continue
        col_words = set(re.split(r"[_\W]+", col.lower()))
        if col_words & q_words:
            candidates.append((df[col].nunique(), col))

    if not candidates:
        # Fallback: if a question word appears in >30% of a column's values,
        # that column is the likely grouping dimension (e.g. "county" → updated_location).
        # Skip columns with ≤1 unique value — they're already a filter, not a group key.
        threshold = max(int(len(df) * 0.3), 1)
        for col in df.columns:
            if not _is_string_col(df[col]):
                continue
            if df[col].nunique() <= 1:
                continue
            sample = df[col].dropna().astype(str).str.lower()
            for word in q_words:
                if len(word) >= 4 and sample.str.contains(word, na=False).sum() >= threshold:
                    candidates.append((df[col].nunique(), col))
                    break

    if candidates:
        candidates.sort()
        return candidates[0][1]
    return None


def _extract_county(series: pd.Series) -> pd.Series:
    def _get(val: str) -> str:
        if pd.isna(val):
            return "Unknown"
        for part in str(val).split(","):
            if "county" in part.lower():
                return part.strip()
        return str(val).strip()
    return series.apply(_get)


# Generic business suffixes that shouldn't be used alone for OEM partial matching
_GENERIC_NAME_WORDS = {
    "corp", "inc", "llc", "ltd", "co", "group", "automotive",
    "manufacturing", "industries", "international", "america", "americas",
    "holdings", "enterprise", "enterprises", "company", "systems",
}

# Relationship-intent keywords → primary_oems is the right filter column
_RELATIONSHIP_KEYWORDS = {
    "linked to", "supplier of", "supply chain", "supplier network",
    "connected to", "supplies to", "works with", "partners with",
}


def _is_relationship_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _RELATIONSHIP_KEYWORDS)


def _expand_partial_value(val: str) -> list[str]:
    """For a compound name like 'Rivian Automotive', extract significant words."""
    words = [w.strip(".,()") for w in val.split()]
    significant = [w for w in words if len(w) >= 4 and w.lower() not in _GENERIC_NAME_WORDS]
    return significant if significant else [val]


# ── Mask builders ─────────────────────────────────────────────────────────────

def _build_col_mask(
    df: pd.DataFrame,
    col: str,
    values: list[str],
    schema_index: dict[str, ColumnMeta],
    allow_word_expansion: bool = False,
) -> pd.Series:
    meta = schema_index.get(col)
    col_mask = pd.Series([False] * len(df), index=df.index)
    for val in values:
        if meta and meta.match_type == "exact":
            col_mask = col_mask | (df[col].astype(str).str.lower() == val.lower())
        else:
            full_mask = df[col].astype(str).str.contains(re.escape(val), case=False, na=False)
            if allow_word_expansion and full_mask.sum() <= 2:
                # Full-value match is too narrow — try significant words OR-ed together
                expanded_mask = pd.Series([False] * len(df), index=df.index)
                for word in _expand_partial_value(val):
                    expanded_mask = expanded_mask | df[col].astype(str).str.contains(
                        re.escape(word), case=False, na=False
                    )
                col_mask = col_mask | expanded_mask
            else:
                col_mask = col_mask | full_mask
    return col_mask


def _build_and_mask(
    df: pd.DataFrame,
    filters: dict[str, list[str]],
    schema_index: dict[str, ColumnMeta],
) -> pd.Series:
    mask = pd.Series([True] * len(df), index=df.index)
    for col, values in filters.items():
        if col not in df.columns:
            continue
        mask = mask & _build_col_mask(df, col, values, schema_index)
    return mask


def _best_single_filter(
    df: pd.DataFrame,
    filters: dict[str, list[str]],
    schema_index: dict[str, ColumnMeta],
    question: str = "",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return the best single-column filter result.

    For relationship queries ("linked to", "supplier of", etc.), prefer
    primary_oems over company so we return suppliers, not the OEM itself.
    """
    is_rel = _is_relationship_query(question)

    # Build scored candidates: (rows, col, values)
    candidates: list[tuple[int, str, pd.DataFrame, dict]] = []
    for col, values in filters.items():
        if col not in df.columns:
            continue
        # In relationship queries, skip company as a primary filter
        if is_rel and col == "company" and "primary_oems" in filters:
            continue
        m = _build_col_mask(df, col, values, schema_index, allow_word_expansion=True)
        candidate = df[m]
        if len(candidate) > 0:
            candidates.append((len(candidate), col, candidate, {col: values}))

    if not candidates:
        # Fall back without the relationship constraint
        for col, values in filters.items():
            if col not in df.columns:
                continue
            m = _build_col_mask(df, col, values, schema_index, allow_word_expansion=True)
            candidate = df[m]
            if len(candidate) > 0:
                candidates.append((len(candidate), col, candidate, {col: values}))

    if candidates:
        # Prefer exact-match column results; among those, pick most selective
        exact_cols = {c for c, meta in schema_index.items() if meta.match_type == "exact"}
        exact_candidates = [(n, col, cdf, cf) for n, col, cdf, cf in candidates if col in exact_cols]
        pool = exact_candidates if exact_candidates else candidates
        pool.sort(key=lambda x: x[0])  # fewest rows = most selective
        _, _, best_df, best_filter = pool[0]
        return best_df, best_filter

    return df, {}


# ── Main retrieve function ────────────────────────────────────────────────────

def retrieve(
    question: str,
    df: pd.DataFrame,
    schema_index: dict[str, ColumnMeta],
    match_result: MatchResult,
) -> RetrievalResult:
    intent = _detect_intent(question)
    filters = match_result.filters

    # Try strict AND across all filters
    and_mask = _build_and_mask(df, filters, schema_index)
    filtered = df[and_mask]
    active_filters = filters

    # Fallback: if AND produced 0 rows, use the most selective individual filter
    if len(filtered) == 0 and filters:
        filtered, active_filters = _best_single_filter(df, filters, schema_index, question)

    # Final fallback: no filters matched → full KB
    if len(filtered) == 0:
        filtered = df
        active_filters = {}

    total_matched = len(filtered)

    # Apply intent-specific operations
    itype = intent["type"]

    if itype == "count":
        result = pd.DataFrame([{"count": total_matched}])

    elif itype == "rank":
        num_col = _numeric_col(filtered, question)
        if num_col and not filtered.empty:
            n = intent.get("n", 1)
            if intent["direction"] == "asc":
                result = filtered.nsmallest(n, num_col)
            else:
                result = filtered.nlargest(n, num_col)
        else:
            result = filtered.head(1)

    elif itype == "aggregate_sum":
        num_col = _numeric_col(filtered, question)
        group_col = _group_col(filtered, question) or _group_col(df, question)

        if num_col and group_col and not filtered.empty:
            if "location" in group_col:
                groupby_series = _extract_county(filtered[group_col])
            else:
                groupby_series = filtered[group_col].astype(str)

            agg = (
                filtered.assign(_group=groupby_series)
                .groupby("_group")[num_col]
                .sum()
                .reset_index()
                .rename(columns={"_group": group_col, num_col: f"total_{num_col}"})
            )
            ascending = intent["direction"] == "asc"
            result = agg.sort_values(f"total_{num_col}", ascending=ascending)
        else:
            result = filtered

    elif itype == "spof":
        role_col = next((c for c in df.columns if "role" in c), None)
        if role_col:
            base = filtered if not filtered.empty else df
            counts = (
                base.groupby(role_col)["company"]
                .count()
                .reset_index()
                .rename(columns={"company": "company_count"})
            )
            spof_roles = counts[counts["company_count"] == 1][role_col].tolist()
            result = base[base[role_col].isin(spof_roles)].copy()
        else:
            result = filtered

    else:
        result = filtered

    support_level = _support_level(total_matched, match_result, active_filters)

    return RetrievalResult(
        rows=result,
        intent=intent,
        total_matched=total_matched,
        filters_applied=active_filters,
        support_level=support_level,
    )


def _support_level(total: int, match_result: MatchResult, active_filters: dict) -> str:
    if total == 0 and not active_filters:
        return "Not Supported by KB"
    if total >= 3 and active_filters and not match_result.unmatched_words:
        return "Fully Supported by KB"
    if total >= 2 and active_filters:
        return "Mostly Supported by KB"
    if total >= 1 and active_filters:
        return "Partially Supported by KB"
    return "Weakly Supported by KB"
