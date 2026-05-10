"""
Deterministic operation detector.

Runs on the ORIGINAL user question (not the rewritten query) to detect
analytical operations that should bypass LLM keyword confidence.

Detected operations:
  - spof              → group by role, count unique companies, find count==1
  - aggregate_sum     → group by dimension, sum numeric metric, rank
  - count             → count rows
  - exhaustive_list   → requires full/untruncated result set

Design rule:
  - Analytical operation phrases are NOT keywords.
  - They must not become term_matcher filters.
  - They override LLM confidence for the detected operation.
"""
from __future__ import annotations

import re
from typing import Any


# ── SPOF detection ────────────────────────────────────────────────────────────

_SPOF_PHRASES = [
    r"single\s+point\s+of\s+failure",
    r"sole\s+supplier",
    r"only\s+one\s+supplier",
    r"only\s+one\s+company",
    r"only\s+a\s+single\s+company",
    r"only\s+a\s+single\s+supplier",
    r"single\s+supplier",
    r"sole\s+provider",
    r"roles?\s+with\s+(?:only\s+)?one\s+company",
    r"roles?\s+(?:served|covered|filled)\s+by\s+(?:only\s+)?(?:a\s+)?single",
    r"roles?\s+(?:served|covered|filled)\s+by\s+(?:only\s+)?one",
]
_SPOF_RE = re.compile(r"\b(?:" + "|".join(_SPOF_PHRASES) + r")\b", re.IGNORECASE)

# ── Aggregate detection ──────────────────────────────────────────────────────

_RANK_HIGH_RE = re.compile(
    r"\b(highest|most|maximum|largest|top|biggest|greatest)\b", re.IGNORECASE
)
_RANK_LOW_RE = re.compile(
    r"\b(lowest|minimum|fewest|least|smallest|bottom)\b", re.IGNORECASE
)
_SUM_RE = re.compile(
    r"\b(total|sum|combined|aggregate|overall)\b", re.IGNORECASE
)
_NUMERIC_TOPIC_RE = re.compile(
    r"\b(employment|employees?|jobs|headcount|investment|amount|capacity)\b",
    re.IGNORECASE,
)
_GROUP_RE = re.compile(
    r"\b(county|counties|city|cities|tier|roles?|category|categories|"
    r"type|types|location|locations)\b",
    re.IGNORECASE,
)

# ── Count detection ───────────────────────────────────────────────────────────

_COUNT_RE = re.compile(
    r"\b(how\s+many|count|number\s+of|total\s+number)\b", re.IGNORECASE
)

# ── Exhaustive list detection ─────────────────────────────────────────────────

_EXHAUSTIVE_RE = re.compile(
    r"\b("
    r"all|every|complete\s+list|full\s+list|entire|"
    r"show\s+all|list\s+all|identify\s+all|provide\s+all|"
    r"how\s+many|count|number\s+of|total|sum|combined|"
    r"highest|lowest|maximum|minimum|most|least|top|bottom|"
    r"single\s+point\s+of\s+failure|sole\s+supplier|only\s+one"
    r")\b",
    re.IGNORECASE,
)

# ── Analytical operation phrases (should never become keywords) ───────────────

_ANALYTICAL_PHRASES_RE = re.compile(
    r"\b("
    r"single\s+point\s+of\s+failure|sole\s+supplier|only\s+one|"
    r"how\s+many|count|number\s+of|total|sum|combined|"
    r"highest|lowest|maximum|minimum|most|least|top|bottom|"
    r"largest|smallest|fewest|greatest|biggest|"
    r"aggregate|overall|average|rank|ranking"
    r")\b",
    re.IGNORECASE,
)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_operation(question: str) -> dict[str, Any]:
    """
    Detect deterministic analytical operation from the original user question.

    Returns a dict with:
      - type: "spof" | "aggregate_sum" | "count" | "exhaustive_list" | "none"
      - requires_exhaustive_retrieval: bool
      - group_by: str (hint, e.g. "EV Supply Chain Role" for SPOF)
      - count_unique: str (hint, e.g. "Company" for SPOF)
      - metric: str (hint, e.g. "Employment" for aggregate)
      - direction: "asc" | "desc" (for aggregate/rank)
      - analytical_phrases: list of phrases that are operations, not keywords
    """
    q = (question or "").strip()

    result: dict[str, Any] = {
        "type": "none",
        "requires_exhaustive_retrieval": False,
        "group_by": None,
        "count_unique": None,
        "metric": None,
        "direction": None,
        "analytical_phrases": [],
    }

    # Collect all analytical phrases found in the question
    analytical = []
    for m in _ANALYTICAL_PHRASES_RE.finditer(q):
        analytical.append(m.group(0).lower().strip())
    result["analytical_phrases"] = list(dict.fromkeys(analytical))  # dedupe

    # Exhaustive retrieval flag
    if _EXHAUSTIVE_RE.search(q):
        result["requires_exhaustive_retrieval"] = True

    # SPOF takes highest priority
    if _SPOF_RE.search(q):
        result["type"] = "spof"
        result["group_by"] = "EV Supply Chain Role"
        result["count_unique"] = "Company"
        result["requires_exhaustive_retrieval"] = True
        return result

    # Count detection
    if _COUNT_RE.search(q):
        result["type"] = "count"
        result["requires_exhaustive_retrieval"] = True
        return result

    # Aggregate sum: needs ranking/sum word + numeric topic + grouping concept
    has_high = bool(_RANK_HIGH_RE.search(q))
    has_low = bool(_RANK_LOW_RE.search(q))
    has_sum = bool(_SUM_RE.search(q))
    has_numeric = bool(_NUMERIC_TOPIC_RE.search(q))
    has_group = bool(_GROUP_RE.search(q))

    if has_group and (has_sum or has_high or has_low) and has_numeric:
        result["type"] = "aggregate_sum"
        result["direction"] = "asc" if has_low else "desc"
        result["requires_exhaustive_retrieval"] = True

        # Detect metric hint
        m = _NUMERIC_TOPIC_RE.search(q)
        if m:
            result["metric"] = m.group(0).capitalize()

        # Detect group_by hint
        g = _GROUP_RE.search(q)
        if g:
            result["group_by"] = g.group(0).capitalize()

        return result

    # Plain rank (highest/lowest without group)
    if has_sum and has_group:
        result["type"] = "aggregate_sum"
        result["direction"] = "asc" if has_low else "desc"
        result["requires_exhaustive_retrieval"] = True
        return result

    # Exhaustive list detection (no specific analytical operation)
    if result["requires_exhaustive_retrieval"] and result["type"] == "none":
        result["type"] = "exhaustive_list"

    return result


def is_analytical_phrase(phrase: str) -> bool:
    """
    Check if a phrase is an analytical operation phrase that should NOT
    be treated as a KB keyword.

    Examples:
      - "single point of failure" → True
      - "highest" → True
      - "Battery Cell" → False
    """
    if not phrase or not phrase.strip():
        return False
    return bool(_ANALYTICAL_PHRASES_RE.fullmatch(phrase.strip()))


def extract_analytical_tokens(question: str) -> set[str]:
    """
    Extract individual tokens from analytical phrases found in the question.

    These tokens should be excluded from keyword/filter matching.
    """
    tokens: set[str] = set()
    for m in _ANALYTICAL_PHRASES_RE.finditer(question or ""):
        for word in m.group(0).lower().split():
            tokens.add(word)
    return tokens
