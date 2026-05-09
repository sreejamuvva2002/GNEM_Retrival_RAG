"""
Query rewriter — fires ONLY when term_matcher detects ambiguous (unmatched) words.

Uses a dedicated local LLM (qwen2.5:14b via Ollama, temperature=0.0) to:
  1. Read the full live KB schema (exact column values, dynamically built — no hardcoding)
  2. Read the system's aggregate intent vocabulary
  3. Rewrite the question replacing ambiguous words with exact KB terms

Always fail-safe: any error or implausible output returns the original question unchanged.
"""
from __future__ import annotations

import requests

from . import config
from .schema_index import ColumnMeta


# ── Aggregate keyword block (intent vocabulary, not data values) ──────────────
# These match the detection sets in retriever._detect_intent exactly.

_AGGREGATE_BLOCK = """\
AGGREGATE KEYWORDS — use these exact phrases for the corresponding intent:
  - Find highest / largest / most:   "highest", "most", "maximum", "largest", "top"
  - Find lowest / smallest / fewest: "lowest", "minimum", "fewest", "least"
  - Sum / total across groups:       "total", "combined", "sum"
  - Count items:                     "how many", "count", "number of"
  - Roles covered by only 1 company: "single point of failure" or "sole supplier"\
"""


# ── Schema context builder ────────────────────────────────────────────────────

def _build_schema_context(schema_index: dict[str, ColumnMeta]) -> str:
    """
    Build a human-readable schema block for the LLM prompt.

    Exact-match columns: list ALL unique values (they are categorical, ≤60 by design).
    Partial-match columns: note as free-text (too many values to list usefully).
    Numeric columns: skipped (not useful for query rewriting).
    Non-filterable columns: skipped.

    Zero hardcoding — column names and values come entirely from the live schema.
    """
    lines: list[str] = []
    for col, meta in schema_index.items():
        if not meta.is_filterable or meta.is_numeric:
            continue
        if meta.match_type == "exact":
            values_str = ", ".join(f'"{v}"' for v in meta.unique_values)
            lines.append(f'  {col} (exact match — use one of these values): {values_str}')
        else:
            lines.append(f'  {col} (free-text search — contains company names / descriptions)')
    return "\n".join(lines)


# ── Prompt builder ────────────────────────────────────────────────────────────

_DO_NOT_REWRITE_BLOCK = """\
DO NOT REWRITE any of these — they are already understood by the system:
  - Aggregate/ranking words: highest, lowest, total, count, most, least, sum, combined,
    how many, number, maximum, minimum, fewest, largest, smallest, top, bottom
  - Geographic/structural words: county, city, location, state, region, area, zone
  - Common question words: which, what, where, who, how, deal, products, components,
    services, companies, suppliers, roles, sector, industry, classified, listed
  - Column-name words: employment, tier, role, category, type, facility\
"""


def _build_prompt(
    question: str,
    schema_context: str,
    unmatched_words: list[str],
) -> str:
    unmatched_str = ", ".join(f'"{w}"' for w in unmatched_words)
    return f"""\
You are a query rewriting assistant for a Georgia EV supply chain knowledge base.

TASK: Rewrite the user's question by replacing ONLY industry-specific technical terms \
that are ambiguous (listed below) with their exact matching values from the KB schema. \
Do not change any other word.

KNOWLEDGE BASE SCHEMA (use ONLY these exact values when rewriting):
{schema_context}

{_AGGREGATE_BLOCK}

{_DO_NOT_REWRITE_BLOCK}

AMBIGUOUS TECHNICAL TERMS DETECTED — replace only those you can confidently map to a KB value:
  {unmatched_str}

ORIGINAL QUESTION:
  {question}

RULES:
  1. Replace ONLY industry-specific technical terms that map to a KB value in the schema.
  2. Use EXACT values from the KB schema (copy character-for-character).
  3. If a term maps to multiple KB values, join them with " or ".
  4. Use the exact aggregate keywords listed above when appropriate.
  5. If you cannot confidently map a term to a KB value, leave it UNCHANGED.
  6. NEVER replace structural, geographic, aggregate, or column-name words.
  7. Output ONLY the rewritten question. No explanation, no prefix, no quotes.

REWRITTEN QUESTION:"""


# ── Ollama call ───────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str:
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.QUERY_REWRITER_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=config.QUERY_REWRITER_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


# ── Sanity check ──────────────────────────────────────────────────────────────

_EXPLANATION_PREFIXES = ("i ", "the question", "rewritten:", "note:", "here", "sure")


def _is_plausible(original: str, rewritten: str) -> bool:
    """Return True only if rewritten looks like a valid rewritten question."""
    if not rewritten:
        return False
    # Reject if the LLM padded wildly (hallucinating extra content)
    if len(rewritten) > len(original) * 3:
        return False
    # Reject if the LLM returned an explanation instead of a question
    low = rewritten.lower()
    if any(low.startswith(p) for p in _EXPLANATION_PREFIXES):
        return False
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def rewrite(
    question: str,
    schema_index: dict[str, ColumnMeta],
    unmatched_words: list[str],
) -> str:
    """
    Rewrite the question using exact KB terminology.

    Returns the original question unchanged if:
      - unmatched_words is empty (nothing ambiguous to fix)
      - QUERY_REWRITER_ENABLED is False
      - The LLM call fails or times out
      - The LLM output fails the plausibility check

    Parameters
    ----------
    question        : original user question
    schema_index    : live KB schema (column metadata with unique values)
    unmatched_words : words from the question that term_matcher could not map to KB values
    """
    if not unmatched_words or not config.QUERY_REWRITER_ENABLED:
        return question

    schema_context = _build_schema_context(schema_index)
    prompt = _build_prompt(question, schema_context, unmatched_words)

    try:
        rewritten = _call_ollama(prompt)
    except Exception:
        return question

    return rewritten if _is_plausible(question, rewritten) else question
