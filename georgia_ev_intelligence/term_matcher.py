"""
Bottom-up term matching: scan the question for KB values rather than
tokenising the question and searching for tokens in the KB.

This ensures zero data hardcoding — all matchable terms come from the live KB.

Fixes included:
  - Robust Tier 1/2, Tier 1 and 2, Tier 1 or 2, Tier 1 & 2 expansion
  - Singular/plural tolerance for words such as supplier/suppliers
  - Safer exact/partial matching
  - Less aggressive redundancy suppression
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema_index import ColumnMeta


@dataclass
class MatchResult:
    filters: dict[str, list[str]] = field(default_factory=dict)
    match_types: dict[str, str] = field(default_factory=dict)  # term → match kind
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
    "including", "such", "into", "not", "no", "full", "set", "complete",
    "entire", "total", "count", "number", "highest", "lowest", "most",
    "least", "top", "bottom",
}

# Minimum character length for a KB value/component to be considered a match.
_MIN_MATCH_LEN = 3


# ── Text normalization helpers ────────────────────────────────────────────────

def _norm_text(text: str) -> str:
    """Lowercase and normalize whitespace."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _tokens(text: str) -> list[str]:
    """Tokenize while preserving slash-style tokens such as 1/2."""
    return re.findall(r"[a-z0-9]+(?:/[a-z0-9]+)?", _norm_text(text))


def _singularize_token(tok: str) -> str:
    """
    Very small singularization helper.

    Enough for supplier/suppliers, companies/company, counties/county.
    Avoids external dependencies.
    """
    tok = tok.lower()

    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _token_set(text: str) -> set[str]:
    toks = _tokens(text)
    out = set(toks)
    out.update(_singularize_token(t) for t in toks)
    return {t for t in out if t}


def _contains_phrase(text: str, phrase: str) -> bool:
    """
    Safer phrase containment with flexible spacing and word boundaries.

    This avoids matching tiny substrings accidentally.
    """
    phrase = str(phrase).strip()
    if not phrase:
        return False

    pattern = r"\b" + r"\s+".join(re.escape(p) for p in phrase.lower().split()) + r"\b"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _deduplicate(values: list[str]) -> list[str]:
    """
    Remove strict prefix duplicates in a same-column match list.
    Example: keep "Tier 1/2", drop "Tier 1" only if it is actually a prefix.
    """
    sorted_desc = sorted(values, key=len, reverse=True)
    result: list[str] = []

    for val in sorted_desc:
        low = val.lower()
        if not any(longer.lower().startswith(low) and longer != val for longer in result):
            result.append(val)

    # Keep stable-ish order by length-desc result, which favors more specific values.
    return result


def _add_match(
    found: list[str],
    match_types: dict[str, str],
    val: str,
    kind: str,
) -> None:
    if val not in found:
        found.append(val)
        match_types[val] = kind


# ── Tier/slash handling ───────────────────────────────────────────────────────

def _extract_requested_tiers(question: str) -> set[str]:
    """
    Extract tier numbers from user shorthand without hardcoding KB values.

    Handles:
      - Tier 1/2
      - Tier 1 / 2
      - Tier 1 and 2
      - Tier 1 or 2
      - Tier 1 & 2
      - Tier 1, 2
      - Tier 1 and Tier 2
    """
    q = _norm_text(question)
    tiers: set[str] = set()

    # Tier 1/2 or tier 1 / 2
    for m in re.finditer(r"\btier\s*(\d+)\s*/\s*(\d+)\b", q):
        tiers.add(m.group(1))
        tiers.add(m.group(2))

    # Tier 1 and/or/&/, 2
    for m in re.finditer(r"\btier\s*(\d+)\s*(?:and|or|&|,)\s*(?:tier\s*)?(\d+)\b", q):
        tiers.add(m.group(1))
        tiers.add(m.group(2))

    # Explicit repeated: Tier 1 ... Tier 2
    for m in re.finditer(r"\btier\s*(\d+)\b", q):
        tiers.add(m.group(1))

    return tiers


def _value_matches_requested_tier(val: str, requested_tiers: set[str]) -> bool:
    """
    Check whether a KB value belongs to any requested tier number.

    This does not hardcode KB values. It only checks whether the KB value itself
    contains "tier <number>" or "tier<number>".
    """
    if not requested_tiers:
        return False

    v = _norm_text(val)

    for tier in requested_tiers:
        if re.search(rf"\btier\s*{re.escape(tier)}\b", v):
            return True

    return False


def _slash_expanded_question(question: str) -> str:
    """
    Build a forgiving string for slash notation:
      "tier 1/2 suppliers" → "tier 1 2 suppliers tier 1 tier 2"
    """
    q = _norm_text(question)
    requested = _extract_requested_tiers(q)
    pieces = [q, re.sub(r"(\w+)\s*/\s*(\w+)", r"\1 \2", q)]

    if requested:
        pieces.extend(f"tier {t}" for t in sorted(requested))

    return " ".join(pieces)


def _value_soft_token_match(question: str, val: str) -> bool:
    """
    Token-level fallback used only for slash/compound notation.

    It allows singular/plural tolerance and ignores generic trailing words
    like supplier/suppliers when the tier/category core already matches.
    """
    q_tokens = _token_set(_slash_expanded_question(question))
    v_tokens = _token_set(val)

    if not v_tokens:
        return False

    # If KB value is tier-like, rely on explicit tier number match.
    requested_tiers = _extract_requested_tiers(question)
    if requested_tiers and _value_matches_requested_tier(val, requested_tiers):
        return True

    # General soft match: all meaningful value tokens are in question tokens.
    # Remove generic entity-class words that often differ singular/plural.
    generic = {
        "supplier", "company", "companies", "manufacturer", "manufacturers",
        "provider", "providers", "facility", "facilities"
    }

    meaningful = {t for t in v_tokens if t not in generic}
    if not meaningful:
        meaningful = v_tokens

    return meaningful.issubset(q_tokens)


# ── Main matcher ──────────────────────────────────────────────────────────────

def match(question: str, schema_index: dict[str, ColumnMeta]) -> MatchResult:
    q_lower = _norm_text(question)
    filters: dict[str, list[str]] = {}
    match_types: dict[str, str] = {}

    requested_tiers = _extract_requested_tiers(question)

    for col, meta in schema_index.items():
        if meta.match_type == "numeric" or not meta.is_filterable:
            continue

        found: list[str] = []

        # Pass 1: full unique values found as phrases in the question.
        for val in meta.unique_values:
            val = str(val)
            if len(val) < _MIN_MATCH_LEN:
                continue

            if _contains_phrase(q_lower, val) or val.lower() in q_lower:
                kind = "exact" if meta.match_type == "exact" else "partial"
                _add_match(found, match_types, val, kind)

        # Pass 1b: explicit tier/slash matching.
        # This fixes "Tier 1/2 suppliers" against KB values like:
        # "Tier 1 Supplier", "Tier 2 Supplier", "Tier 1/2", etc.
        if requested_tiers:
            for val in meta.unique_values:
                val = str(val)
                if len(val) < _MIN_MATCH_LEN or val in found:
                    continue

                if _value_matches_requested_tier(val, requested_tiers):
                    kind = "exact" if meta.match_type == "exact" else "partial"
                    _add_match(found, match_types, val, f"tier_{kind}")

        # Pass 1c: slash/compound soft token fallback.
        # Useful when exact value contains singular/plural differences.
        if "/" in question or requested_tiers:
            for val in meta.unique_values:
                val = str(val)
                if len(val) < _MIN_MATCH_LEN or val in found:
                    continue

                if _value_soft_token_match(question, val):
                    kind = "exact" if meta.match_type == "exact" else "partial"
                    _add_match(found, match_types, val, f"soft_{kind}")

        # Pass 2: location/name components
        for comp in getattr(meta, "components", []):
            comp = str(comp)
            if len(comp) < _MIN_MATCH_LEN:
                continue

            if (_contains_phrase(q_lower, comp) or comp.lower() in q_lower) and comp not in found:
                _add_match(found, match_types, comp, "component")

        if found:
            found = _deduplicate(found)
            filters[col] = found

    # Redundancy suppression:
    # Suppress partial-column values only when they are clearly redundant with an
    # exact-column value. Do not delete whole useful columns because one value
    # overlaps an exact value.
    exact_surface = set()
    for col, vals in filters.items():
        meta = schema_index.get(col)
        if meta and meta.match_type == "exact":
            for v in vals:
                exact_surface.add(v.lower())

    for col in list(filters.keys()):
        meta = schema_index.get(col)

        if not meta or meta.match_type != "partial":
            continue

        non_redundant = []
        for v in filters[col]:
            v_low = v.lower()

            # Keep equal values and values with meaningful independent context.
            # Remove only proper-substring duplicates.
            is_redundant = any(
                v_low in exact_val and v_low != exact_val
                for exact_val in exact_surface
            )

            if not is_redundant:
                non_redundant.append(v)

        if non_redundant:
            filters[col] = non_redundant
        else:
            del filters[col]

    # Collect question words not matched by any filter value.
    q_words = [w.strip(".,;?!\"'()") for w in question.split()]
    matched_surface = {v.lower() for vals in filters.values() for v in vals}

    unmatched = [
        w for w in q_words
        if w.lower() not in _STOPWORDS
        and len(w) >= _MIN_MATCH_LEN
        and not any(w.lower() in m for m in matched_surface)
    ]

    return MatchResult(
        filters=filters,
        match_types=match_types,
        unmatched_words=unmatched,
    )