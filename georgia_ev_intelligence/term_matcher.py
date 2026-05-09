"""
Bottom-up term matching: scan the question for KB values rather than
tokenising the question and searching for tokens in the KB.

This ensures zero data hardcoding — all matchable terms come from the live KB.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from .schema_index import ColumnMeta


@dataclass
class MatchResult:
    filters: dict[str, list[str]] = field(default_factory=dict)
    match_types: dict[str, str] = field(default_factory=dict)    # term → match kind
    unmatched_words: list[str] = field(default_factory=list)


_STOPWORDS = {
    "in", "the", "and", "or", "a", "an", "of", "to", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "which", "what", "where", "who", "how", "when", "all", "show", "list",
    "give", "find", "tell", "me", "their", "its", "with", "for", "by",
    "from", "on", "at", "that", "this", "these", "those", "each", "any",
    "only", "also", "both", "georgia", "company", "companies", "supplier",
    "suppliers", "linked", "related", "associated", "across", "among",
    "including", "such", "into", "not", "no", "full", "set",
}

# Minimum character length for a KB value to be considered a match.
_MIN_MATCH_LEN = 3


def _deduplicate(values: list[str]) -> list[str]:
    """Remove values that are strict prefixes of a longer value in the same list."""
    sorted_desc = sorted(values, key=len, reverse=True)
    result: list[str] = []
    for val in sorted_desc:
        if not any(longer.lower().startswith(val.lower()) and longer != val for longer in result):
            result.append(val)
    return result


def match(question: str, schema_index: dict[str, ColumnMeta]) -> MatchResult:
    q_lower = question.lower()
    filters: dict[str, list[str]] = {}
    match_types: dict[str, str] = {}

    for col, meta in schema_index.items():
        if meta.match_type == "numeric" or not meta.is_filterable:
            continue

        found: list[str] = []

        # Pass 1: full unique values found as substrings in the question
        for val in meta.unique_values:
            if len(val) < _MIN_MATCH_LEN:
                continue
            if val.lower() in q_lower:
                found.append(val)
                match_types[val] = "exact" if meta.match_type == "exact" else "partial"

        # Pass 2: location components (e.g. "Troup County" from "West Point, Troup County")
        for comp in meta.components:
            if len(comp) < _MIN_MATCH_LEN:
                continue
            if comp.lower() in q_lower and comp not in found:
                found.append(comp)
                match_types[comp] = "component"

        if found:
            # Remove prefix duplicates (e.g. keep "Tier 1/2", drop "Tier 1")
            found = _deduplicate(found)
            filters[col] = found

    # Suppress partial-column matches that are substrings of already-matched exact values.
    # E.g. "thermal" in product_service is redundant when "Thermal Management" already
    # matched in ev_supply_chain_role (exact).
    exact_surface = set()
    for col, vals in filters.items():
        meta = schema_index.get(col)
        if meta and meta.match_type == "exact":
            for v in vals:
                exact_surface.add(v.lower())

    for col in list(filters.keys()):
        meta = schema_index.get(col)
        if meta and meta.match_type == "partial":
            non_redundant = [
                v for v in filters[col]
                if not any(v.lower() in exact_val for exact_val in exact_surface)
            ]
            if non_redundant:
                filters[col] = non_redundant
            else:
                del filters[col]

    # Collect question words not matched by any filter value
    q_words = [w.strip(".,;?!\"'()") for w in question.split()]
    matched_surface = {v.lower() for vals in filters.values() for v in vals}
    unmatched = [
        w for w in q_words
        if w.lower() not in _STOPWORDS
        and len(w) >= _MIN_MATCH_LEN
        and not any(w.lower() in m for m in matched_surface)
    ]

    return MatchResult(filters=filters, match_types=match_types, unmatched_words=unmatched)
