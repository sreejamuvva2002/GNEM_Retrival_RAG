"""
Applies matched filters to the KB DataFrame and handles aggregate / rank intents.
All operations are pure pandas — no external DB required.

Fallback cascade when strict AND produces 0 rows:
  1. Try strict AND of all filters
  2. If 0 rows: try each individual column filter, pick most selective (fewest rows)
  3. If still 0: return full DataFrame with a low-confidence path

Important:
  - pipeline.py is responsible for passing deterministic/full bases for
    analytical intents.
  - apply_intent() should not run aggregate analytics on RRF-truncated evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

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

_RANK_HIGH = {"highest", "most", "maximum", "largest", "top", "biggest", "greatest"}
_RANK_LOW = {"lowest", "minimum", "fewest", "least", "smallest", "bottom"}
_SUM_WORDS = {"total", "sum", "combined", "aggregate", "overall"}
_COUNT_WORDS = {"how many", "count", "number of", "total number"}
_SPOF_WORDS = {
    "single point of failure",
    "sole supplier",
    "only one company",
    "only a single",
    "sole provider",
    "single supplier",
}

_GROUP_WORDS = {
    "county",
    "counties",
    "city",
    "cities",
    "tier",
    "role",
    "roles",
    "category",
    "categories",
    "type",
    "types",
    "location",
    "locations",
}

_NUMERIC_TOPIC_WORDS = {
    "employment",
    "employees",
    "employee",
    "jobs",
    "headcount",
    "investment",
    "amount",
    "capacity",
    "count",
    "number",
}


def _word_in(phrase: str, text: str) -> bool:
    """Whole-word match to avoid 'count' matching 'county'."""
    return bool(re.search(r"\b" + re.escape(phrase) + r"\b", text))


def _contains_any(words: set[str], text: str) -> bool:
    return any(_word_in(w, text) for w in words)


def _detect_intent(question: str) -> dict:
    """
    Detect simple analytical intent.

    Important correction:
    - "Which county has the highest employment?" should be aggregate_sum,
      not rank, because it requires groupby county + sum employment.
    """
    q = (question or "").lower()

    if any(phrase in q for phrase in _SPOF_WORDS):
        return {"type": "spof"}

    for phrase in _COUNT_WORDS:
        if _word_in(phrase, q):
            return {"type": "count"}

    has_high = _contains_any(_RANK_HIGH, q)
    has_low = _contains_any(_RANK_LOW, q)
    has_sum = _contains_any(_SUM_WORDS, q)
    has_group = _contains_any(_GROUP_WORDS, q)
    has_numeric_topic = _contains_any(_NUMERIC_TOPIC_WORDS, q)

    # Grouped numeric ranking/sum: county with highest employment, role with
    # highest total employment, tier with lowest total jobs, etc.
    if has_group and (has_sum or has_high or has_low) and has_numeric_topic:
        direction = "asc" if has_low else "desc"
        return {"type": "aggregate_sum", "direction": direction}

    # Original aggregate wording: total/sum by county/tier/role.
    if has_sum and has_group:
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
    """
    Select numeric column relevant to the question.

    Prefer numeric columns whose name overlaps the question. Fall back to
    common employment/investment-like columns before arbitrary numeric columns.
    """
    if df is None or df.empty:
        return None

    q_words = set(re.split(r"\W+", (question or "").lower()))
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols:
        return None

    for col in numeric_cols:
        col_words = set(re.split(r"[_\W]+", col.lower()))
        if q_words & col_words:
            return col

    # Prefer common business numeric columns.
    preferred_patterns = [
        "employment",
        "employees",
        "employee",
        "jobs",
        "headcount",
        "investment",
        "amount",
        "capacity",
    ]
    for pattern in preferred_patterns:
        for col in numeric_cols:
            if pattern in col.lower():
                return col

    # Fallback: first numeric col that is not an ID.
    non_id = [
        c for c in numeric_cols
        if "id" not in c.lower() and c != "_row_id"
    ]
    return non_id[0] if non_id else numeric_cols[0]


def _group_col(df: pd.DataFrame, question: str) -> str | None:
    """
    Select grouping column.

    Prefer explicit column-name overlap. For county questions, prefer a column
    containing county, otherwise location-like columns from which county can be
    extracted.
    """
    if df is None or df.empty:
        return None

    q = (question or "").lower()
    q_words = set(re.split(r"\W+", q))
    candidates: list[tuple[int, int, str]] = []

    # Strong preference for county/location when user asks county.
    if _word_in("county", q) or _word_in("counties", q):
        for col in df.columns:
            if not _is_string_col(df[col]):
                continue
            low = col.lower()
            if "county" in low:
                return col
        for col in df.columns:
            if not _is_string_col(df[col]):
                continue
            low = col.lower()
            if "location" in low or "city" in low or "address" in low:
                return col

    for col in df.columns:
        if not _is_string_col(df[col]):
            continue
        col_words = set(re.split(r"[_\W]+", col.lower()))
        overlap = len(q_words & col_words)
        if overlap:
            # lower nunique = more grouping-like; higher overlap = better
            candidates.append((-overlap, df[col].nunique(), col))

    if candidates:
        candidates.sort()
        return candidates[0][2]

    # Fallback: if a question word appears in >30% of a column's values,
    # that column is likely a grouping dimension.
    threshold = max(int(len(df) * 0.3), 1)
    fallback_candidates: list[tuple[int, str]] = []

    for col in df.columns:
        if not _is_string_col(df[col]):
            continue
        if df[col].nunique() <= 1:
            continue

        sample = df[col].dropna().astype(str).str.lower()
        for word in q_words:
            if len(word) >= 4 and sample.str.contains(re.escape(word), na=False).sum() >= threshold:
                fallback_candidates.append((df[col].nunique(), col))
                break

    if fallback_candidates:
        fallback_candidates.sort()
        return fallback_candidates[0][1]

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


# Generic business suffixes that should not be used alone for OEM partial matching.
_GENERIC_NAME_WORDS = {
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

# Relationship-intent keywords → primary_oems is the right filter column.
_RELATIONSHIP_KEYWORDS = {
    "linked to",
    "supplier of",
    "supply chain",
    "supplier network",
    "connected to",
    "supplies to",
    "works with",
    "partners with",
}


def _is_relationship_query(question: str) -> bool:
    q = (question or "").lower()
    return any(kw in q for kw in _RELATIONSHIP_KEYWORDS)


def _expand_partial_value(val: str) -> list[str]:
    """For a compound name like 'Rivian Automotive', extract significant words."""
    words = [w.strip(".,()") for w in str(val).split()]
    significant = [
        w for w in words
        if len(w) >= 4 and w.lower() not in _GENERIC_NAME_WORDS
    ]
    return significant if significant else [str(val)]


# ── Mask builders ─────────────────────────────────────────────────────────────

def _build_col_mask(
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
                for word in _expand_partial_value(val):
                    expanded_mask = expanded_mask | series.str.contains(
                        re.escape(word),
                        case=False,
                        na=False,
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
    """AND across columns, OR within each column's values."""
    mask = pd.Series([True] * len(df), index=df.index)

    for col, values in (filters or {}).items():
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
    """
    Return the best single-column filter result.

    For relationship queries ("linked to", "supplier of", etc.), prefer
    primary_oems over company so we return suppliers, not the OEM itself.
    """
    is_rel = _is_relationship_query(question)

    candidates: list[tuple[int, int, str, pd.DataFrame, dict[str, list[str]]]] = []

    for col, values in (filters or {}).items():
        if col not in df.columns:
            continue

        if is_rel and col == "company" and "primary_oems" in filters:
            continue

        m = _build_col_mask(
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

            m = _build_col_mask(
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


# ── Main retrieve function ────────────────────────────────────────────────────

def retrieve(
    question: str,
    df: pd.DataFrame,
    schema_index: dict[str, ColumnMeta],
    match_result: MatchResult,
) -> RetrievalResult:
    intent = _detect_intent(question)
    filters = match_result.filters

    # Try strict AND across all filters.
    and_mask = _build_and_mask(df, filters, schema_index)
    filtered = df[and_mask]
    active_filters = filters

    # Fallback: if AND produced 0 rows, use the most selective individual filter.
    if len(filtered) == 0 and filters:
        filtered, active_filters = _best_single_filter(
            df,
            filters,
            schema_index,
            question,
        )

    # Final fallback: no filters matched → full KB.
    if len(filtered) == 0:
        filtered = df
        active_filters = {}

    total_matched = len(filtered)

    # Delegate intent-specific transformation.
    result, intent = apply_intent(filtered, question, df, intent=intent)

    support_level = _support_level(
        total_matched,
        match_result.unmatched_words,
        active_filters,
    )

    return RetrievalResult(
        rows=result,
        intent=intent,
        total_matched=total_matched,
        filters_applied=active_filters,
        support_level=support_level,
    )


# ── Standalone intent application ─────────────────────────────────────────────

def apply_intent(
    filtered: pd.DataFrame,
    question: str,
    full_df: pd.DataFrame,
    intent: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply intent-specific transformation to an already-filtered DataFrame.

    Parameters
    ----------
    filtered:
        Pre-filtered DataFrame. For analytical questions, pipeline.py should
        pass a deterministic full/filtered KB base, not top-k retrieval evidence.
    question:
        Original question string.
    full_df:
        Complete KB DataFrame.
    intent:
        Optional pre-detected intent.
    """
    if intent is None:
        intent = _detect_intent(question)

    itype = intent["type"]

    if itype == "count":
        return pd.DataFrame([{"count": len(filtered)}]), intent

    elif itype == "rank":
        rank_base = filtered if filtered is not None and not filtered.empty else full_df
        num_col = _numeric_col(rank_base, question)

        if num_col and not rank_base.empty:
            n = int(intent.get("n", 1))
            if intent["direction"] == "asc":
                result = rank_base.nsmallest(n, num_col)
            else:
                result = rank_base.nlargest(n, num_col)
        else:
            result = rank_base.head(1)

        return result, intent

    elif itype == "aggregate_sum":
        # Corrected: use the deterministic filtered base from pipeline.py.
        # Do NOT blindly use full_df here, otherwise scoped questions such as
        # "among Tier 1/2 suppliers" will ignore filters.
        agg_base = filtered if filtered is not None and not filtered.empty else full_df

        num_col = _numeric_col(agg_base, question)
        group_col = _group_col(agg_base, question)

        if num_col and group_col and not agg_base.empty:
            if "location" in group_col.lower() or "address" in group_col.lower():
                groupby_series = _extract_county(agg_base[group_col])
            else:
                groupby_series = agg_base[group_col].astype(str)

            agg = (
                agg_base.assign(_group=groupby_series)
                .groupby("_group", dropna=False)[num_col]
                .sum()
                .reset_index()
                .rename(columns={"_group": group_col, num_col: f"total_{num_col}"})
            )

            ascending = intent["direction"] == "asc"
            return agg.sort_values(f"total_{num_col}", ascending=ascending), intent

        return filtered, intent

    elif itype == "spof":
        role_col = next((c for c in full_df.columns if "role" in c.lower()), None)

        if role_col and "company" in full_df.columns:
            # SPOF must count across the full KB, not retrieval subset.
            base = full_df.dropna(subset=[role_col]).copy()

            counts = (
                base.groupby(role_col)["company"]
                .nunique()
                .reset_index()
                .rename(columns={"company": "company_count"})
            )

            spof_roles = counts[counts["company_count"] == 1][role_col].tolist()
            result = base[base[role_col].isin(spof_roles)].copy()

            # Attach company_count for transparency.
            if not result.empty:
                result = result.merge(counts, on=role_col, how="left")

            return result, intent

        return filtered, intent

    else:
        return filtered, intent


def _support_level(total: int, unmatched_words: list[str], active_filters: dict) -> str:
    if total == 0 and not active_filters:
        return "Not Supported by KB"
    if total >= 3 and active_filters and not unmatched_words:
        return "Fully Supported by KB"
    if total >= 2 and active_filters:
        return "Mostly Supported by KB"
    if total >= 1 and active_filters:
        return "Partially Supported by KB"
    return "Weakly Supported by KB"