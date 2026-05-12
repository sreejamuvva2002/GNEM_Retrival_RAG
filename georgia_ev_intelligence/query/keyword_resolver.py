"""
Deterministic keyword resolver.

Runs BEFORE the Stage 2 query rewriter to establish ground-truth about what
the user's question directly matches in the live KB.

Three categories of keyword resolution:

  1. PERFECT keywords — a live KB value directly matches a span in the user
     question.  These become deterministic filters.
  2. CANDIDATE keywords — approximate/expanded matches that may help retrieval
     but are NOT deterministic filters.
  3. REJECTED keywords — column names, analytical operation phrases, or values
     from incompatible columns.

Zero hardcoding:
  Everything is derived dynamically from schema_index[col].unique_values,
  column metadata, and compatibility rules.

Design:
  User question
  → deterministic live KB string scanner
  → perfect keyword resolver
  → candidate keyword resolver
  → operation detector
  → term_matcher / retrieval / dataframe logic
  → final LLM formats answer only from evidence
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ..data.schema import ColumnMeta
from .operation_detector import is_analytical_phrase
from .term_matcher import (
    _norm_text,
    _contains_phrase,
    _normalise_for_comparison,
    _extract_question_ngrams,
    _is_tier_compatible_column,
    _extract_requested_tiers,
    _value_matches_requested_tier,
    _extract_slash_phrases,
    _MIN_MATCH_LEN,
)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class ResolvedKeyword:
    """A single resolved keyword match."""
    value: str               # The exact live KB value
    column: str              # Schema column it came from
    match_type: str          # "exact", "normalised", "contains", "tier_expanded", "soft"
    user_span: str           # The user question span that triggered the match
    score: float             # Higher is better; exact > normalised > contains > tier > soft
    reason: str = ""         # Why it was placed in this category


@dataclass
class KeywordResolution:
    """
    Complete keyword resolution result.

    perfect_keywords:   exact live KB value matches → deterministic filters
    candidate_keywords: approximate matches → retrieval only, NOT filters
    rejected_keywords:  column names / analytical ops / incompatible → excluded
    """
    perfect_keywords: list[ResolvedKeyword] = field(default_factory=list)
    candidate_keywords: list[ResolvedKeyword] = field(default_factory=list)
    rejected_keywords: list[ResolvedKeyword] = field(default_factory=list)

    # Deterministic filters derived from perfect keywords only.
    # Maps column → list of exact live KB values.
    deterministic_filters: dict[str, list[str]] = field(default_factory=dict)

    # Summary flags
    has_perfect: bool = False
    has_candidates_only: bool = False

    def to_debug_dict(self) -> dict:
        """Compact debug representation."""
        return {
            "perfect": [
                {"value": k.value, "col": k.column, "type": k.match_type}
                for k in self.perfect_keywords
            ],
            "candidate": [
                {"value": k.value, "col": k.column, "type": k.match_type, "reason": k.reason}
                for k in self.candidate_keywords
            ],
            "rejected": [
                {"value": k.value, "reason": k.reason}
                for k in self.rejected_keywords
            ],
            "deterministic_filters": self.deterministic_filters,
            "has_perfect": self.has_perfect,
            "has_candidates_only": self.has_candidates_only,
        }


# ── Column name detection ─────────────────────────────────────────────────────

def _norm_key(name: str) -> str:
    """Normalise a column name for comparison."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _is_column_name(term: str, schema_index: dict[str, ColumnMeta]) -> bool:
    """Check whether *term* is a column name or normalised column key."""
    if not term or not term.strip():
        return False
    low = term.strip().lower()
    norm = _norm_key(term)
    for col in schema_index:
        if low == col.lower() or norm == _norm_key(col):
            return True
    return False


# ── Column compatibility ──────────────────────────────────────────────────────

_TIER_COMPATIBLE_PATTERNS = frozenset({
    "category", "tier", "supplier_type",
    "supplier_or_affiliation_type", "affiliation", "classification",
})

_PRODUCT_COMPATIBLE_PATTERNS = frozenset({
    "ev_supply_chain_role", "role", "product_service", "product",
    "service", "industry_group", "primary_facility_type",
})

_LOCATION_COMPATIBLE_PATTERNS = frozenset({
    "location", "updated_location", "county", "city", "state",
    "address", "region",
})


def _classify_phrase_type(phrase: str) -> str:
    """Classify a user phrase as tier / product_component / location / analytical / general."""
    if not phrase or not phrase.strip():
        return "general"
    p = phrase.strip().lower()

    if re.search(r"\btier\s*\d", p):
        return "tier"
    if is_analytical_phrase(phrase):
        return "analytical"
    if re.search(
        r"\b(battery\s+cell|battery\s+pack|thermal\s+management|power\s+electronics|"
        r"charging\s+infrastructure|battery\s+module|electric\s+motor|ev\s+component|"
        r"wiring\s+harness)\b", p,
    ):
        return "product_component"
    if re.search(r"\b(county|city|location|georgia|state|region)\b", p) and len(p.split()) <= 3:
        return "location"
    return "general"


def _is_column_compatible(phrase_type: str, col: str) -> bool:
    """Validate that a phrase type is compatible with a specific column."""
    norm = _norm_key(col)

    if phrase_type == "tier":
        return any(p in norm for p in _TIER_COMPATIBLE_PATTERNS)
    if phrase_type == "product_component":
        return any(p in norm for p in _PRODUCT_COMPATIBLE_PATTERNS)
    if phrase_type == "location":
        return any(p in norm for p in _LOCATION_COMPATIBLE_PATTERNS)
    if phrase_type == "analytical":
        return False  # analytical phrases are never column filters

    # "general" phrases have no restriction
    return True


# ── Core resolver ─────────────────────────────────────────────────────────────

def resolve_keywords(
    question: str,
    schema_index: dict[str, ColumnMeta],
    *,
    max_ngram: int = 6,
) -> KeywordResolution:
    """
    Deterministic keyword resolution from user question against live KB values.

    This runs BEFORE the LLM rewriter and establishes ground truth about
    which user phrases directly match live KB values.

    Steps:
      1. Extract n-grams from the question (longest first).
      2. For each n-gram, scan every filterable column's unique_values for
         exact, normalised, or contains matches.
      3. Score matches: exact > normalised > contains.
      4. Also run tier expansion for tier-compatible columns.
      5. Classify each match as perfect / candidate / rejected.
      6. Apply slash/compound conflict resolution: if an exact match exists
         for a slash phrase, suppress expanded partial matches.
      7. Build deterministic_filters from perfect keywords only.

    Returns KeywordResolution with all three categories populated.
    """
    q_lower = question.lower()
    q_norm = _normalise_for_comparison(question)
    ngrams = _extract_question_ngrams(question, max_ngram=max_ngram)
    ngram_lower_set = {ng.lower() for ng in ngrams}
    ngram_norm_set = {_normalise_for_comparison(ng) for ng in ngrams}

    requested_tiers = _extract_requested_tiers(question)
    slash_phrases = _extract_slash_phrases(question)

    # Collect all raw matches with scoring
    raw_matches: list[ResolvedKeyword] = []

    for col, meta in schema_index.items():
        if meta.match_type == "numeric" or not meta.is_filterable:
            continue

        for val in meta.unique_values:
            val_str = str(val).strip()
            if len(val_str) < _MIN_MATCH_LEN:
                continue

            val_lower = val_str.lower()
            val_norm = _normalise_for_comparison(val_str)

            best_type = None
            best_span = ""
            best_score = 0.0

            # Tier 1: exact match (case-insensitive, question ngram == KB value)
            if val_lower in ngram_lower_set:
                best_type = "exact"
                best_span = val_lower
                best_score = 3.0 + len(val_str) / 100.0

            # Tier 2: normalised match (slash/space collapsed)
            if best_type is None and val_norm in ngram_norm_set:
                best_type = "normalised"
                best_span = val_norm
                best_score = 2.0 + len(val_str) / 100.0

            # Tier 3: KB value appears as phrase in the question
            if best_type is None:
                if _contains_phrase(q_lower, val_str) or val_lower in q_lower:
                    best_type = "contains"
                    best_span = val_lower
                    best_score = 1.0 + len(val_str) / 100.0

            # Tier 4: tier expansion match (only for tier-compatible columns)
            if (
                best_type is None
                and requested_tiers
                and _is_tier_compatible_column(col)
                and _value_matches_requested_tier(val_str, requested_tiers)
            ):
                best_type = "tier_expanded"
                best_span = f"tier expansion from: {', '.join(sorted(requested_tiers))}"
                best_score = 0.5 + len(val_str) / 100.0

            if best_type:
                raw_matches.append(ResolvedKeyword(
                    value=val_str,
                    column=col,
                    match_type=best_type,
                    user_span=best_span,
                    score=best_score,
                ))

    # Sort by score descending
    raw_matches.sort(key=lambda m: -m.score)

    # ── Classify into perfect / candidate / rejected ──────────────────────

    # Build anchor set for slash conflict resolution
    anchor_values: set[str] = set()   # exact-match values for slash phrases
    anchor_norms: set[str] = set()

    if slash_phrases:
        for m in raw_matches:
            if m.match_type in ("exact", "normalised"):
                v_lower = m.value.lower()
                v_norm = _normalise_for_comparison(m.value)
                for sp in slash_phrases:
                    sp_norm = _normalise_for_comparison(sp)
                    if v_lower == sp or v_norm == sp_norm or sp in v_lower or sp_norm in v_norm:
                        anchor_values.add(m.value)
                        anchor_norms.add(v_norm)

    perfect: list[ResolvedKeyword] = []
    candidate: list[ResolvedKeyword] = []
    rejected: list[ResolvedKeyword] = []

    # Track which user spans have already been covered by a perfect match
    # to avoid duplicate perfect matches for overlapping spans.
    covered_spans: set[str] = set()

    for m in raw_matches:
        # ── Rejection checks ──

        # Reject: column name used as value
        if _is_column_name(m.value, schema_index):
            m.reason = "column_name_not_a_value"
            rejected.append(m)
            continue

        # Reject: analytical operation phrase
        if is_analytical_phrase(m.value):
            m.reason = "analytical_operation_phrase"
            rejected.append(m)
            continue

        # Classify phrase type and check column compatibility
        phrase_type = _classify_phrase_type(m.user_span if m.match_type != "tier_expanded" else m.value)
        if not _is_column_compatible(phrase_type, m.column):
            m.reason = f"incompatible_{phrase_type}_column={m.column}"
            rejected.append(m)
            continue

        # ── Slash conflict resolution ──

        # If anchors exist for slash phrases, suppress expanded matches
        if anchor_values and m.value not in anchor_values:
            if m.match_type == "tier_expanded":
                m.reason = "suppressed_by_exact_slash_anchor"
                candidate.append(m)
                continue

            # Check if this value is a substring of an anchor
            v_norm = _normalise_for_comparison(m.value)
            is_substring_of_anchor = any(
                v_norm in anorm and v_norm != anorm
                for anorm in anchor_norms
            )
            if is_substring_of_anchor:
                m.reason = "substring_of_exact_slash_anchor"
                candidate.append(m)
                continue

        # ── Perfect vs Candidate classification ──

        if m.match_type in ("exact", "normalised", "contains"):
            # Direct KB value match → PERFECT
            span_key = f"{m.column}:{m.user_span}"
            if span_key not in covered_spans:
                covered_spans.add(span_key)
                m.reason = "direct_live_kb_value_match"
                perfect.append(m)
            else:
                m.reason = "duplicate_span_already_covered"
                candidate.append(m)
        elif m.match_type == "tier_expanded":
            # Tier expansion without an exact anchor → CANDIDATE
            m.reason = "tier_expansion_no_exact_anchor"
            candidate.append(m)
        else:
            m.reason = "approximate_match"
            candidate.append(m)

    # ── Build deterministic filters from perfect keywords ─────────────────

    det_filters: dict[str, list[str]] = {}
    seen_values: set[str] = set()  # Dedupe across columns

    for pk in perfect:
        key = f"{pk.column}:{pk.value}"
        if key not in seen_values:
            seen_values.add(key)
            det_filters.setdefault(pk.column, []).append(pk.value)

    result = KeywordResolution(
        perfect_keywords=perfect,
        candidate_keywords=candidate,
        rejected_keywords=rejected,
        deterministic_filters=det_filters,
        has_perfect=len(perfect) > 0,
        has_candidates_only=len(perfect) == 0 and len(candidate) > 0,
    )

    return result
