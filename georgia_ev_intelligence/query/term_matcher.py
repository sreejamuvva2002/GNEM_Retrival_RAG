"""
Bottom-up term matching: scan the question for KB values rather than
tokenising the question and searching for tokens in the KB.

This ensures zero data hardcoding — all matchable terms come from the live KB.

Fixes included:
  - Robust Tier 1/2, Tier 1 and 2, Tier 1 or 2, Tier 1 & 2 expansion
  - Singular/plural tolerance for words such as supplier/suppliers
  - Safer exact/partial matching
  - Less aggressive redundancy suppression
  - Exact-value precedence: if a user phrase exactly matches a live KB value
    in a compatible column, prefer that value over expanded partial components
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ..data.schema import ColumnMeta


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


# ── Column-compatibility helpers ──────────────────────────────────────────────

def _is_tier_compatible_column(col: str) -> bool:
    """
    Check whether a column is appropriate for tier/classification filtering.

    Tier phrases (Tier 1/2, Tier 1 and 2, etc.) should only create filters
    in columns that represent tiers, categories, or classifications — NOT in
    role, product, company, location, or OEM columns.
    """
    norm = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")
    return any(kw in norm for kw in (
        "category",
        "tier",
        "supplier_type",
        "supplier_or_affiliation_type",
        "affiliation",
        "classification",
    ))


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


# ── Generic live-value matching with exact-value precedence ───────────────────

def _normalise_for_comparison(text: str) -> str:
    """Normalise text for value comparison: lowercase, collapse whitespace/slashes."""
    return re.sub(r"[\s/]+", " ", str(text).lower()).strip()


def _extract_question_ngrams(question: str, max_ngram: int = 6) -> list[str]:
    """
    Extract word n-grams from the question for matching against live KB values.

    Preserves slash notation (e.g. "1/2") as single tokens and generates
    n-grams from 1 to max_ngram words, longest first.
    """
    # Tokenize preserving slashes inside words
    tokens = re.findall(r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)*", question)
    if not tokens:
        return []

    ngrams: list[str] = []
    for n in range(min(max_ngram, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if len(phrase) >= _MIN_MATCH_LEN:
                ngrams.append(phrase)
    return ngrams


def find_best_live_value_matches(
    question: str,
    schema_index: dict[str, ColumnMeta],
    *,
    compatible_column_predicate: Callable[[str], bool] | None = None,
    max_ngram: int = 6,
    top_n: int = 10,
) -> list[dict]:
    """
    Find the best live KB value matches for phrases in the question.

    Scoring tiers (highest wins):
      1. exact_match    — question ngram exactly equals a live KB value (case-insensitive)
      2. normalised_match — normalised form matches (slash/space collapsed)
      3. contains_match  — KB value is contained in the question as a phrase

    When an exact slash/compound value exists (e.g. "Tier 1/2"), its component
    expansions ("Tier 1", "Tier 2") are suppressed by _resolve_slash_conflicts().

    Parameters
    ----------
    question : str
        The user question.
    schema_index : dict[str, ColumnMeta]
        Live schema with unique_values per column.
    compatible_column_predicate : callable, optional
        If provided, only consider columns where predicate(col) is True.
    max_ngram : int
        Maximum n-gram length to extract from the question.
    top_n : int
        Maximum number of matches to return.

    Returns
    -------
    List of dicts with keys:
      - value: str           (the exact live KB value)
      - column: str          (the column it came from)
      - match_tier: str      ("exact_match", "normalised_match", "contains_match")
      - score: float         (higher is better)
      - ngram: str           (the question phrase that matched)
    """
    q_lower = question.lower()
    q_norm = _normalise_for_comparison(question)
    ngrams = _extract_question_ngrams(question, max_ngram=max_ngram)

    # Build lookup sets for fast matching
    ngram_lower_set = {ng.lower() for ng in ngrams}
    ngram_norm_set = {_normalise_for_comparison(ng) for ng in ngrams}

    matches: list[dict] = []

    for col, meta in schema_index.items():
        if meta.match_type == "numeric" or not meta.is_filterable:
            continue
        if compatible_column_predicate and not compatible_column_predicate(col):
            continue

        for val in meta.unique_values:
            val_str = str(val).strip()
            if len(val_str) < _MIN_MATCH_LEN:
                continue

            val_lower = val_str.lower()
            val_norm = _normalise_for_comparison(val_str)

            best_tier = None
            best_ngram = None
            best_score = 0.0

            # Tier 1: exact match (case-insensitive)
            if val_lower in ngram_lower_set:
                best_tier = "exact_match"
                best_ngram = val_lower
                # Score: longer matches are better
                best_score = 3.0 + len(val_str) / 100.0

            # Tier 2: normalised match (slash/space collapsed)
            if best_tier is None and val_norm in ngram_norm_set:
                best_tier = "normalised_match"
                best_ngram = val_norm
                best_score = 2.0 + len(val_str) / 100.0

            # Tier 3: KB value appears as phrase in question
            if best_tier is None:
                if _contains_phrase(q_lower, val_str) or val_lower in q_lower:
                    best_tier = "contains_match"
                    best_ngram = val_lower
                    best_score = 1.0 + len(val_str) / 100.0

            if best_tier:
                matches.append({
                    "value": val_str,
                    "column": col,
                    "match_tier": best_tier,
                    "score": best_score,
                    "ngram": best_ngram,
                })

    # Sort by score descending, then by value length descending (more specific first)
    matches.sort(key=lambda m: (-m["score"], -len(m["value"])))
    return matches[:top_n]


def _extract_slash_phrases(question: str) -> list[str]:
    """
    Extract slash-containing phrases from the question.

    E.g. "Tier 1/2 suppliers" → ["tier 1/2"]
         "Battery Cell/Pack" → ["battery cell/pack"]
    """
    # Match patterns like: <words> <digit/digit> or <word/word>
    q = _norm_text(question)
    phrases: list[str] = []

    # Pattern: optional prefix word(s) + slash token
    # e.g. "tier 1/2", "cell/pack", "1/2"
    for m in re.finditer(r"((?:[a-z]+\s+)?\d+/\d+|[a-z]+/[a-z]+)", q):
        phrases.append(m.group(0).strip())

    return phrases


def _resolve_slash_conflicts(
    found: list[str],
    match_types: dict[str, str],
    col: str,
    schema_index: dict[str, ColumnMeta],
    question: str,
) -> list[str]:
    """
    If a slash/compound user phrase exactly matches a live KB value in this
    column, suppress expanded partial matches that conflict.

    Example:
      User: "Tier 1/2"
      KB values: ["Tier 1/2", "Tier 1", "Tier 2", "Tier 2/3"]
      found = ["Tier 1/2", "Tier 1", "Tier 2", "Tier 2/3"]

      After resolution:
      found = ["Tier 1/2"]

    The rule is:
      - If the user's slash phrase (e.g. "tier 1/2") exactly/normalise-matches
        a KB value, that value is the "anchor".
      - All other found values that were matched only via tier-expansion
        (match_type starts with "tier_" or "soft_") are removed.
      - Values matched via Pass 1 exact/contains phrase match are kept ONLY
        if they are not strict substrings of the anchor.

    This is fully data-driven: no KB values are hardcoded.
    """
    if not found or "/" not in question:
        return found

    meta = schema_index.get(col)
    if not meta:
        return found

    slash_phrases = _extract_slash_phrases(question)
    if not slash_phrases:
        return found

    # Find anchor values: live KB values that exactly/normalise-match a slash phrase
    anchors: set[str] = set()
    anchor_norms: set[str] = set()

    for val in found:
        val_lower = val.lower()
        val_norm = _normalise_for_comparison(val)
        for sp in slash_phrases:
            sp_norm = _normalise_for_comparison(sp)
            # Check: KB value matches the user's slash phrase
            if val_lower == sp or val_norm == sp_norm:
                anchors.add(val)
                anchor_norms.add(val_norm)
            # Also check: KB value contains the slash phrase as a core
            # (e.g. "Tier 1/2 Supplier" contains "tier 1/2")
            elif sp in val_lower or sp_norm in val_norm:
                anchors.add(val)
                anchor_norms.add(val_norm)

    if not anchors:
        return found

    # We have anchors — suppress expanded partial matches
    resolved: list[str] = []
    for val in found:
        if val in anchors:
            # Always keep anchors
            resolved.append(val)
            continue

        mtype = match_types.get(val, "")

        # Expanded tier/soft matches are suppressed when an anchor exists
        if mtype.startswith("tier_") or mtype.startswith("soft_"):
            continue

        # For Pass 1 exact/partial matches, keep only if the value is NOT
        # a component substring of any anchor
        val_norm = _normalise_for_comparison(val)
        is_component_of_anchor = any(
            val_norm in anorm and val_norm != anorm
            for anorm in anchor_norms
        )
        if is_component_of_anchor:
            continue

        resolved.append(val)

    return resolved if resolved else found


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
        # IMPORTANT: Only apply tier matching in tier-compatible columns.
        # Without this guard, "Tier 1 automotive components" in
        # EV Supply Chain Role would be selected as a tier filter.
        if requested_tiers and _is_tier_compatible_column(col):
            for val in meta.unique_values:
                val = str(val)
                if len(val) < _MIN_MATCH_LEN or val in found:
                    continue

                if _value_matches_requested_tier(val, requested_tiers):
                    kind = "exact" if meta.match_type == "exact" else "partial"
                    _add_match(found, match_types, val, f"tier_{kind}")

        # Pass 1c: slash/compound soft token fallback.
        # Useful when exact value contains singular/plural differences.
        # For tier-related queries, restrict soft matching to tier-compatible
        # columns. For non-tier slash queries, allow general soft matching.
        if "/" in question or requested_tiers:
            # If this is a tier query and the column is NOT tier-compatible,
            # skip soft tier matching to avoid polluting role/product columns.
            if requested_tiers and not _is_tier_compatible_column(col):
                pass  # Skip tier-driven soft matching in non-tier columns
            else:
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
            # Resolve slash/compound conflicts: if the user's slash phrase
            # exactly matches a live KB value, suppress expanded partials.
            found = _resolve_slash_conflicts(
                found, match_types, col, schema_index, question,
            )
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
