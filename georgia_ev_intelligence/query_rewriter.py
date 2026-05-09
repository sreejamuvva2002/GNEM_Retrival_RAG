"""
Two-stage KB-grounded query rewriter.

Stage 1 — Semantic Probe Generation
  DeepSeek generates 5-10 diverse semantic probes from the user question and
  schema metadata (column names / types) only.  No KB data values are shown.
  Output is tagged "semantic_only_not_kb_verified" to prevent downstream code
  from treating it as ground truth.

Stage 2 — KB-Grounded Query Rewrite
  After probe retrieval and dynamic KB term extraction, DeepSeek rewrites the
  question using ONLY terms that appeared in the retrieved candidate rows.
  It must not invent domain vocabulary or generate a final answer.

Both stages call the same local Ollama endpoint (DeepSeek-R1-Distill-Qwen-7B).
Every LLM call strips <think>...</think> blocks, validates JSON, and retries
on parse failure before returning None (which triggers a safe fallback).

The legacy `rewrite()` function is preserved for backward compatibility but is
no longer called by the main pipeline.
"""
from __future__ import annotations

import json
import re
import requests
import pandas as pd

from . import config
from .schema_index import ColumnMeta


# ── Required JSON keys for each stage ────────────────────────────────────────

_STAGE1_REQUIRED = frozenset({
    "intent",
    "target_columns",
    "semantic_probes",
    "explicit_filters",
    "requires_broad_retrieval",
    "stage",
    "status",
})

_STAGE2_REQUIRED = frozenset({
    "stage",
    "intent",
    "explicit_filters",
    "target_columns",
    "mapped_user_phrases",
    "final_rewritten_queries",
    "negative_queries_or_terms_to_avoid",
    "requires_exhaustive_retrieval",
    "confidence",
    "warnings",
})

_VALID_INTENTS = frozenset({
    "list", "count", "compare", "group",
    "supplier_discovery", "location", "relationship", "yes_no", "other",
})

_VALID_CONFIDENCE = frozenset({"high", "medium", "low"})


# ── Schema context builders ───────────────────────────────────────────────────

def _build_stage1_schema_context(schema_index: dict[str, ColumnMeta]) -> str:
    """Column names and match-types only — no KB data values shown to Stage 1."""
    lines: list[str] = []
    for col, meta in schema_index.items():
        if not meta.is_filterable:
            continue
        if meta.is_numeric:
            type_str = "numeric"
        elif meta.match_type == "exact":
            type_str = "categorical"
        else:
            type_str = "free-text"
        lines.append(f"  - {col} ({type_str})")
    return "\n".join(lines)


def _build_schema_context(schema_index: dict[str, ColumnMeta]) -> str:
    """
    Full schema context with sample values — used by Stage 2 for column grounding.
    Exact columns: all unique values listed.
    Partial columns: up to 80 sample values listed.
    """
    lines: list[str] = []
    for col, meta in schema_index.items():
        if not meta.is_filterable or meta.is_numeric:
            continue
        if meta.match_type == "exact":
            vals = ", ".join(f'"{v}"' for v in meta.unique_values)
            lines.append(f'  {col} (categorical — exact values): {vals}')
        else:
            cap = 80
            sample = meta.unique_values[:cap]
            vals = ", ".join(f'"{v}"' for v in sample)
            if len(meta.unique_values) > cap:
                vals += f" … ({len(meta.unique_values)} total)"
            lines.append(f'  {col} (free-text — partial match): {vals}')
    return "\n".join(lines)


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_stage1_prompt(user_question: str, schema_ctx: str) -> str:
    return f"""\
You are a semantic query decomposer for a structured knowledge base retrieval system.
Generate diverse semantic probe queries from the user question and schema metadata only.
Do NOT generate an answer. Do NOT use domain terms not present in the question or schema.

KNOWLEDGE BASE COLUMNS:
{schema_ctx}

USER QUESTION:
{user_question}

TASK:
1. Generate 5-10 diverse semantic probe queries (broad, medium, narrow angles).
2. Identify query intent: list | count | compare | group | supplier_discovery | location | relationship | yes_no | other
3. List target_columns from the schema most relevant to the question.
4. Extract explicit_filters: any location, company, tier, or category constraint stated directly in the question, mapped to a column name.
5. Set requires_broad_retrieval=true if the question is vague or covers a wide topic.
6. Do NOT include company names, product names, or technology terms not in the question or column names.
7. Do NOT answer the question.

OUTPUT: a valid JSON object only. No explanation, no markdown fences, no preamble.

{{
  "intent": "<list|count|compare|group|supplier_discovery|location|relationship|yes_no|other>",
  "target_columns": ["<col_name>"],
  "semantic_probes": ["<probe_1>", "<probe_2>", "<probe_3>", "<probe_4>", "<probe_5>"],
  "explicit_filters": {{"<col_name>": "<value_from_question>"}},
  "requires_broad_retrieval": true,
  "stage": "semantic_probe_generation",
  "status": "semantic_only_not_kb_verified"
}}"""


def _build_stage2_prompt(
    user_question: str,
    schema_ctx: str,
    semantic_probes: list[str],
    kb_terms: dict,
    explicit_filters: dict,
) -> str:
    probes_str    = json.dumps(semantic_probes, indent=2)
    terms_str     = json.dumps(kb_terms.get("kb_discovered_terms", []), indent=2)
    sources_brief = [
        {"term": s["term"], "source_columns": s["source_columns"]}
        for s in kb_terms.get("term_sources", [])[:20]
    ]
    sources_str   = json.dumps(sources_brief, indent=2)
    filters_str   = json.dumps(explicit_filters, indent=2)

    return f"""\
You are a KB-grounded query rewriter for a structured knowledge base retrieval system.
Rewrite the user question into precise retrieval queries using ONLY the discovered KB terms below.
Do NOT invent terms. Do NOT generate a final answer.

KNOWLEDGE BASE COLUMNS:
{schema_ctx}

USER QUESTION:
{user_question}

SEMANTIC PROBES (Stage 1 — for context only, not final):
{probes_str}

DYNAMICALLY DISCOVERED KB TERMS (from retrieved candidate rows — use ONLY these):
{terms_str}

TERM SOURCES (column origin of each term):
{sources_str}

EXPLICIT FILTERS DETECTED:
{filters_str}

RULES:
1. Write 3-5 final_rewritten_queries using ONLY: original question words, column names, or kb_discovered_terms above.
2. Do NOT use any term not present in kb_discovered_terms or the original question.
3. Do NOT list company names or produce a final answer.
4. If kb_discovered_terms are weak or unrelated, set confidence="low" and explain in warnings.
5. Set requires_exhaustive_retrieval=true if the question contains: all, every, complete list, how many, total, count.
6. For each mapped_user_phrase: assign confidence high (direct KB match), medium (related), or low (inferred).
7. Include negative_queries_or_terms_to_avoid to reduce false positives.

OUTPUT: a valid JSON object only. No explanation, no markdown fences, no preamble.

{{
  "stage": "kb_grounded_query_rewrite",
  "intent": "<intent>",
  "explicit_filters": {{}},
  "target_columns": [],
  "mapped_user_phrases": [
    {{"user_phrase": "<phrase>", "kb_supported_terms": [], "mapping_source": "kb_discovered_terms", "confidence": "<high|medium|low>"}}
  ],
  "final_rewritten_queries": ["<query_1>", "<query_2>", "<query_3>"],
  "negative_queries_or_terms_to_avoid": [],
  "requires_exhaustive_retrieval": false,
  "confidence": "<high|medium|low>",
  "warnings": []
}}"""


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str:
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.QUERY_REWRITER_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 1024,
            },
        },
        timeout=config.QUERY_REWRITER_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw


# ── JSON parsing and validation ───────────────────────────────────────────────

def _parse_and_validate_json(
    raw: str,
    required_keys: frozenset[str],
) -> dict | None:
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if not required_keys.issubset(data.keys()):
        return None
    return data


def _call_with_retry(
    prompt: str,
    required_keys: frozenset[str],
) -> dict | None:
    current = prompt
    for _ in range(config.MAX_REWRITER_RETRIES):
        try:
            raw = _call_ollama(current)
        except Exception:
            return None
        result = _parse_and_validate_json(raw, required_keys)
        if result is not None:
            return result
        current = prompt + "\n\nIMPORTANT: Output ONLY a valid JSON object. No other text."
    return None


# ── Fallback builder ──────────────────────────────────────────────────────────

def build_fallback_stage2(
    user_question: str,
    stage1: dict,
    kb_terms: dict,
) -> dict:
    """
    Safe fallback when Stage 2 fails or KB evidence is too sparse.
    Uses Stage 1 probes (first 3) + original question as rewritten queries.
    """
    probes = stage1.get("semantic_probes", [])
    fallback_queries = [user_question] + probes[:3]
    discovered_count = len(kb_terms.get("kb_discovered_terms", []))
    warnings = [
        "insufficient_kb_terms_fallback_to_stage1_probes",
        f"discovered_terms_count={discovered_count}",
    ]
    return {
        "stage": "kb_grounded_query_rewrite",
        "intent": stage1.get("intent", "other"),
        "explicit_filters": stage1.get("explicit_filters", {}),
        "target_columns": stage1.get("target_columns", []),
        "mapped_user_phrases": [],
        "final_rewritten_queries": fallback_queries,
        "negative_queries_or_terms_to_avoid": [],
        "requires_exhaustive_retrieval": False,
        "confidence": "low",
        "warnings": warnings,
    }


# ── Retrieval quality scoring ─────────────────────────────────────────────────

def score_retrieval(
    candidates_df: pd.DataFrame,
    explicit_filters: dict,
    kb_terms: dict,
) -> dict:
    """
    Score the probe retrieval quality.  Returns {"weak": bool, ...signals}.
    "weak" triggers fallback: prepend original question to rewritten queries.
    """
    unique_rows       = len(candidates_df["_row_id"].unique()) if "_row_id" in candidates_df.columns else len(candidates_df)
    avg_score         = float(candidates_df["_score"].mean()) if "_score" in candidates_df.columns else 0.0
    discovered_terms  = len(kb_terms.get("kb_discovered_terms", []))

    # Check whether any explicit filter value appears in candidates
    filter_hits = 0
    if explicit_filters:
        for col, val in explicit_filters.items():
            if col in candidates_df.columns:
                hit = candidates_df[col].astype(str).str.lower().str.contains(
                    str(val).lower(), na=False
                ).any()
                if hit:
                    filter_hits += 1

    weak = (
        unique_rows < config.PROBE_MIN_ROWS
        or discovered_terms < config.KB_TERM_MIN_DISCOVERED
        or (bool(explicit_filters) and filter_hits == 0)
    )

    return {
        "unique_row_count": unique_rows,
        "avg_dense_similarity": round(avg_score, 4),
        "discovered_term_count": discovered_terms,
        "explicit_filter_hits": filter_hits,
        "weak": weak,
    }


# ── Public two-stage API ──────────────────────────────────────────────────────

def stage1_probe_generation(
    user_question: str,
    schema_index: dict[str, ColumnMeta],
) -> dict | None:
    """
    Stage 1: Generate semantic probes from the user question and schema
    metadata only.  Returns a validated dict or None on failure.
    """
    if not config.QUERY_REWRITER_ENABLED:
        return None

    schema_ctx = _build_stage1_schema_context(schema_index)
    prompt     = _build_stage1_prompt(user_question, schema_ctx)
    result     = _call_with_retry(prompt, _STAGE1_REQUIRED)

    if result is None:
        return None

    # Normalise: ensure semantic_probes is a non-empty list of strings
    probes = result.get("semantic_probes", [])
    if not isinstance(probes, list) or not probes:
        return None
    result["semantic_probes"] = [str(p) for p in probes if str(p).strip()]

    # Normalise intent
    if result.get("intent") not in _VALID_INTENTS:
        result["intent"] = "other"

    # Normalise explicit_filters to dict[str, str]
    ef = result.get("explicit_filters", {})
    result["explicit_filters"] = {str(k): str(v) for k, v in ef.items()} if isinstance(ef, dict) else {}

    return result


def stage2_kb_grounded_rewrite(
    user_question: str,
    schema_index: dict[str, ColumnMeta],
    stage1: dict,
    kb_terms: dict,
    explicit_filters: dict,
) -> dict | None:
    """
    Stage 2: Rewrite the user question using dynamically discovered KB terms.
    Returns a validated dict or None on failure (triggers build_fallback_stage2).
    """
    if not config.QUERY_REWRITER_ENABLED:
        return None

    schema_ctx = _build_schema_context(schema_index)
    probes     = stage1.get("semantic_probes", [])
    prompt     = _build_stage2_prompt(
        user_question, schema_ctx, probes, kb_terms, explicit_filters
    )
    result = _call_with_retry(prompt, _STAGE2_REQUIRED)

    if result is None:
        return None

    # Normalise final_rewritten_queries
    queries = result.get("final_rewritten_queries", [])
    if not isinstance(queries, list) or not queries:
        return None
    result["final_rewritten_queries"] = [str(q) for q in queries if str(q).strip()]

    # Normalise confidence
    if result.get("confidence") not in _VALID_CONFIDENCE:
        result["confidence"] = "low"

    # Ensure list fields exist
    result.setdefault("warnings", [])
    result.setdefault("negative_queries_or_terms_to_avoid", [])
    result.setdefault("mapped_user_phrases", [])

    return result


# ── Legacy API (kept for backward compatibility) ──────────────────────────────

_AGGREGATE_BLOCK = """\
AGGREGATE KEYWORDS — use these exact phrases for the corresponding intent:
  - Find highest / largest / most:   "highest", "most", "maximum", "largest", "top"
  - Find lowest / smallest / fewest: "lowest", "minimum", "fewest", "least"
  - Sum / total across groups:       "total", "combined", "sum"
  - Count items:                     "how many", "count", "number of"
  - Roles covered by only 1 company: "single point of failure" or "sole supplier"\
"""

_DO_NOT_REWRITE_BLOCK = """\
DO NOT REWRITE any of these — they are already understood by the system:
  - Aggregate/ranking words: highest, lowest, total, count, most, least, sum, combined,
    how many, number, maximum, minimum, fewest, largest, smallest, top, bottom
  - Geographic/structural words: county, city, location, state, region, area, zone
  - Common question words: which, what, where, who, how, deal, products, components,
    services, companies, suppliers, roles, sector, industry, classified, listed
  - Column-name words: employment, tier, role, category, type, facility\
"""

_EXPLANATION_PREFIXES = ("i ", "the question", "rewritten:", "note:", "here", "sure")


def _is_plausible(original: str, rewritten: str) -> bool:
    if not rewritten:
        return False
    if len(rewritten) > len(original) * 3:
        return False
    low = rewritten.lower()
    return not any(low.startswith(p) for p in _EXPLANATION_PREFIXES)


def rewrite(
    question: str,
    schema_index: dict[str, ColumnMeta],
    unmatched_words: list[str],
) -> str:
    """Legacy single-pass rewriter (not used by the two-stage pipeline)."""
    if not unmatched_words or not config.QUERY_REWRITER_ENABLED:
        return question

    schema_ctx = _build_schema_context(schema_index)
    unmatched_str = ", ".join(f'"{w}"' for w in unmatched_words)
    prompt = f"""\
You are a query rewriting assistant for a Georgia EV supply chain knowledge base.

TASK: Rewrite the user's question by replacing ONLY the ambiguous terms listed below \
with their exact matching values from the KB schema. Do not change any other word.

KNOWLEDGE BASE SCHEMA (use ONLY these exact values when rewriting):
{schema_ctx}

{_AGGREGATE_BLOCK}

{_DO_NOT_REWRITE_BLOCK}

AMBIGUOUS TECHNICAL TERMS DETECTED:
  {unmatched_str}

ORIGINAL QUESTION:
  {question}

RULES:
  1. Replace ONLY terms that map to a KB value.
  2. Use EXACT values from the schema (copy character-for-character).
  3. If a term maps to multiple KB values, join with " or ".
  4. Leave terms unchanged if you cannot confidently map them.
  5. Output ONLY the rewritten question. No explanation, no quotes.

REWRITTEN QUESTION:"""

    try:
        rewritten = _call_ollama(prompt)
    except Exception:
        return question

    return rewritten if _is_plausible(question, rewritten) else question