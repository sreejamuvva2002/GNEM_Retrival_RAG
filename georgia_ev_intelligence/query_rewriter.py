from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import requests

from . import config
from .schema_index import ColumnMeta


# ──────────────────────────────────────────────────────────────────────────────
# Two-stage KB-grounded query rewriter
#
# Stage 1: semantic probe generation
#   - Input: user question + metadata only
#   - Output: broad/medium/narrow probes used to discover KB vocabulary
#   - Important: probes are semantic guesses, NOT KB facts
#
# Stage 2: KB-grounded rewrite
#   - Input: user question + metadata + dynamically discovered KB terms
#   - Output: retrieval-ready rewritten queries using only:
#       original question words, metadata/column names, explicit filters,
#       and kb_discovered_terms
#
# This file intentionally keeps the old public function names/signatures.
# ──────────────────────────────────────────────────────────────────────────────


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

_EXHAUSTIVE_RE = re.compile(
    r"\b("
    r"all|every|complete|complete\s+list|full\s+list|entire|"
    r"how\s+many|count|number\s+of|total|sum|combined|"
    r"highest|lowest|maximum|minimum|most|least|top|bottom|"
    r"single\s+point\s+of\s+failure|sole\s+supplier|only\s+one"
    r")\b",
    re.IGNORECASE,
)

_COUNT_RE = re.compile(r"\b(how\s+many|count|number\s+of|total|sum|combined)\b", re.IGNORECASE)
_GROUP_RE = re.compile(r"\b(group|by county|by city|by role|by tier|across|highest|lowest|most|least)\b", re.IGNORECASE)

# Words allowed inside rewritten queries even if not in KB terms.
# These are retrieval scaffolding words, not domain facts.
_GENERIC_QUERY_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "company",
    "companies", "complete", "count", "county", "counties", "each", "every",
    "for", "from", "give", "has", "have", "highest", "how", "in", "include",
    "including", "is", "list", "location", "locations", "map", "most", "of",
    "only", "or", "provide", "related", "role", "roles", "show", "single",
    "supplier", "suppliers", "their", "the", "them", "to", "total", "under",
    "what", "which", "with", "within",
    # common retrieval operators / field words
    "filter", "match", "matches", "query", "search",
})

# Minimal metadata descriptions based on column names only.
# These are not KB values; they are schema/metadata hints.
_DEFAULT_COLUMN_HINTS = {
    "company": "Company or organization name. Use for explicit company/entity constraints only.",
    "companies": "Company or organization name. Use for explicit company/entity constraints only.",
    "supplier": "Supplier/company entity field, not a supplier tier by itself.",
    "suppliers": "Supplier/company entity field, not a supplier tier by itself.",
    "category": "Supplier classification, affiliation type, or tier-like category. Use for tier/classification questions.",
    "supplier_type": "Supplier classification or tier-like category. Use for tier/classification questions.",
    "supplier_or_affiliation_type": "Supplier classification, affiliation type, or tier-like category. Use for tier/classification questions.",
    "tier": "Supplier tier/classification metadata. Use for Tier 1, Tier 2, Tier 1/2-style constraints when such terms are in the user question.",
    "ev_supply_chain_role": "EV supply-chain role/category. Use for role-based filters, grouping, and role coverage questions.",
    "role": "Role/category field. Use for role-based filters and grouping.",
    "product_service": "Product, service, component, material, or capability text. Use for product/service/component matching.",
    "product": "Product, service, component, material, or capability text.",
    "service": "Product, service, component, material, or capability text.",
    "primary_oems": "OEM/customer relationship field. Use for questions asking linked OEMs or customers; do not use this as a location filter.",
    "oem": "OEM/customer relationship field. Use for questions asking linked OEMs or customers.",
    "updated_location": "Normalized place such as city/county/region. Use for location display or city/county filters. If the user only says Georgia and the KB is Georgia-scoped, do not force exact match to the text Georgia.",
    "location": "Place such as city/county/region/state. Use for geographic filters and display.",
    "county": "County field. Use for county grouping, county filtering, and county-level aggregation.",
    "city": "City field. Use for city filtering and city-level grouping.",
    "state": "State field. Use for state-level filtering if present.",
    "employment": "Numeric employment/headcount field. Use for employment totals, sums, rankings, and comparisons.",
    "employees": "Numeric employment/headcount field. Use for employment totals, sums, rankings, and comparisons.",
    "employee_count": "Numeric employment/headcount field. Use for employment totals, sums, rankings, and comparisons.",
    "primary_facility_type": "Facility type metadata. Use for plant/facility questions. Do not use as supplier tier unless the user explicitly asks facility type.",
    "facility_type": "Facility type metadata. Use for plant/facility questions. Do not use as supplier tier unless the user explicitly asks facility type.",
    "notes": "Free-text evidence/details field. Use as supporting context only.",
    "description": "Free-text evidence/details field. Use as supporting context only.",
    "source": "Source/reference metadata. Do not use as semantic product or role evidence.",
}


# ── Small utilities ───────────────────────────────────────────────────────────

def _cfg(name: str, default: Any) -> Any:
    """Read optional config values without breaking older config.py files."""
    return getattr(config, name, default)


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)?", str(text).lower())


def _dedupe_preserve(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for item in items:
        key = str(item).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _as_str_list(value: Any, max_items: int | None = None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = [str(v) for v in value if str(v).strip()]
    else:
        items = [str(value)]
    cleaned = _dedupe_preserve([v.strip() for v in items if v.strip()])
    return cleaned[:max_items] if max_items else cleaned


def _is_exhaustive_question(question: str) -> bool:
    return bool(_EXHAUSTIVE_RE.search(question or ""))


def _looks_like_georgia_scope_filter(value: Any) -> bool:
    return str(value).strip().lower() in {"georgia", "ga", "state of georgia"}


def _schema_columns(schema_index: dict[str, ColumnMeta]) -> list[str]:
    return list(schema_index.keys())


def _schema_lookup(schema_index: dict[str, ColumnMeta]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for col in schema_index:
        lookup[_norm_key(col)] = col
        lookup[str(col).lower()] = col
    return lookup


def _normalise_column_name(col: Any, schema_index: dict[str, ColumnMeta]) -> str | None:
    if col is None:
        return None
    raw = str(col).strip()
    if not raw:
        return None
    lookup = _schema_lookup(schema_index)
    return lookup.get(_norm_key(raw)) or lookup.get(raw.lower())


def _normalise_target_columns(value: Any, schema_index: dict[str, ColumnMeta]) -> list[str]:
    cols: list[str] = []
    for col in _as_str_list(value):
        mapped = _normalise_column_name(col, schema_index)
        if mapped:
            cols.append(mapped)
    return _dedupe_preserve(cols)


def _normalise_explicit_filters(
    filters: Any,
    schema_index: dict[str, ColumnMeta],
) -> dict[str, str]:
    """
    Keep filters conservative.

    Important:
    - Do not force "Georgia" into a city/county location exact filter unless the
      schema has a true state-like column and the model selected it.
    - Drop hallucinated column names instead of passing them downstream.
    """
    if not isinstance(filters, dict):
        return {}

    out: dict[str, str] = {}
    for raw_col, raw_val in filters.items():
        val = str(raw_val).strip()
        if not val:
            continue

        mapped_col = _normalise_column_name(raw_col, schema_index)

        # If the model returns dataset_scope or scope, keep it as metadata-only.
        # Downstream exact-filter code should ignore keys beginning with "__".
        if mapped_col is None and _norm_key(raw_col) in {"dataset_scope", "scope", "kb_scope"}:
            out["__dataset_scope"] = val
            continue

        if mapped_col is None:
            continue

        # If user says "Georgia" and the column is a city/county/location-like
        # field, treat it as dataset scope to avoid exact filtering to "Georgia"
        # when rows contain "Atlanta, Fulton County", "West Point", etc.
        norm_col = _norm_key(mapped_col)
        if _looks_like_georgia_scope_filter(val) and any(x in norm_col for x in ("location", "county", "city")):
            out["__dataset_scope"] = val
            continue

        out[mapped_col] = val

    return out


def _column_description(col: str) -> str:
    """
    Build metadata-only descriptions from the column name.
    No KB values are used here.
    """
    # Optional project-level descriptions can be supplied in config.py:
    # COLUMN_DESCRIPTIONS = {"category": "...", "ev_supply_chain_role": "..."}
    custom = _cfg("COLUMN_DESCRIPTIONS", {})
    if isinstance(custom, dict):
        if col in custom:
            return str(custom[col])
        low = str(col).lower()
        if low in custom:
            return str(custom[low])
        norm = _norm_key(col)
        if norm in custom:
            return str(custom[norm])

    norm = _norm_key(col)
    if norm in _DEFAULT_COLUMN_HINTS:
        return _DEFAULT_COLUMN_HINTS[norm]

    # Fallback by substring.
    for key, desc in _DEFAULT_COLUMN_HINTS.items():
        if key and key in norm:
            return desc

    return "No additional description available. Use only if the user wording clearly targets this field."


# ── Schema context builders ───────────────────────────────────────────────────

def _build_metadata_schema_context(schema_index: dict[str, ColumnMeta]) -> str:
    """
    Metadata-only schema context. No KB values, examples, or unique values.
    Safe for both Stage 1 and Stage 2.
    """
    lines: list[str] = []
    for col, meta in schema_index.items():
        if not getattr(meta, "is_filterable", True):
            continue

        if getattr(meta, "is_numeric", False):
            type_str = "numeric"
        elif getattr(meta, "match_type", "") == "exact":
            type_str = "categorical"
        else:
            type_str = "free-text"

        match_type = getattr(meta, "match_type", "partial")
        desc = _column_description(col)
        lines.append(
            f'  - "{col}" | type={type_str} | match_type={match_type} | description={desc}'
        )

    return "\n".join(lines) if lines else "  (No filterable columns found.)"


def _build_stage1_schema_context(schema_index: dict[str, ColumnMeta]) -> str:
    """Backward-compatible name. Metadata only."""
    return _build_metadata_schema_context(schema_index)


def _build_schema_context(schema_index: dict[str, ColumnMeta]) -> str:
    """
    Legacy full schema context with sample values.

    Kept only for the old legacy `rewrite()` function.
    The two-stage pipeline intentionally does NOT use this because Stage 2 must
    be grounded only in dynamically discovered KB terms.
    """
    lines: list[str] = []
    for col, meta in schema_index.items():
        if not getattr(meta, "is_filterable", True) or getattr(meta, "is_numeric", False):
            continue
        if getattr(meta, "match_type", "") == "exact":
            vals = ", ".join(f'"{v}"' for v in getattr(meta, "unique_values", []))
            lines.append(f'  {col} (categorical — exact values): {vals}')
        else:
            cap = int(_cfg("QUERY_REWRITER_SCHEMA_VALUE_CAP", 80))
            values = list(getattr(meta, "unique_values", []))
            sample = values[:cap]
            vals = ", ".join(f'"{v}"' for v in sample)
            if len(values) > cap:
                vals += f" … ({len(values)} total)"
            lines.append(f'  {col} (free-text — partial match): {vals}')
    return "\n".join(lines)


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_stage1_prompt(user_question: str, schema_ctx: str) -> str:
    return f"""\
You are Stage 1 of a two-stage KB-grounded query rewriter.

ROLE:
Generate SEMANTIC PROBE QUERIES for retrieval exploration.
You are NOT answering the question.
You are NOT claiming any term exists in the KB.
Your probes will be validated later by retrieval.

IMPORTANT:
You see only metadata/column descriptions, not KB values.
You MAY use general semantic understanding to create exploratory probes.
Those probes are semantic guesses only and must be tagged "semantic_only_not_kb_verified".
Do NOT invent company names.
Do NOT produce SQL.
Do NOT produce a final answer.
Do NOT map user phrases to final KB terms in this stage.

KNOWLEDGE BASE METADATA:
{schema_ctx}

USER QUESTION:
{user_question}

TASK:
1. Detect intent: list | count | compare | group | supplier_discovery | location | relationship | yes_no | other.
2. Identify target_columns from the metadata.
3. Generate 5-10 diverse semantic_probes:
   - include the original wording as one angle,
   - include broad probes,
   - include medium probes,
   - include narrow probes,
   - preserve explicit user constraints such as location, company, tier, role, product/service, OEM, county, count/list.
4. Extract explicit_filters only when the user directly states a constraint.
5. If the user says "Georgia" and there is no true State column, treat Georgia as dataset scope, not as an exact city/county/location value.
6. Set requires_broad_retrieval=true for vague, semantic, list, count, group, all/every/complete/how many/total questions.

OUTPUT RULES:
Return a valid JSON object only.
No markdown.
No code fences.
No explanation.
No SQL.

EXPECTED JSON:
{{
  "intent": "<list|count|compare|group|supplier_discovery|location|relationship|yes_no|other>",
  "target_columns": ["<schema_column_name>"],
  "semantic_probes": [
    "<original-or-close-user-wording>",
    "<broad semantic probe>",
    "<medium semantic probe>",
    "<narrow semantic probe>",
    "<alternate semantic probe>"
  ],
  "explicit_filters": {{"<schema_column_name_or___dataset_scope>": "<value_from_question>"}},
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
    probes_str = json.dumps(_as_str_list(semantic_probes, max_items=12), indent=2)
    discovered_terms = _as_str_list(kb_terms.get("kb_discovered_terms", []), max_items=80)
    terms_str = json.dumps(discovered_terms, indent=2)

    sources_brief: list[dict[str, Any]] = []
    for s in kb_terms.get("term_sources", [])[:40]:
        if not isinstance(s, dict):
            continue
        sources_brief.append({
            "term": str(s.get("term", "")),
            "source_columns": _as_str_list(s.get("source_columns", []), max_items=6),
        })

    sources_str = json.dumps(sources_brief, indent=2)
    filters_str = json.dumps(explicit_filters or {}, indent=2)

    return f"""\
You are Stage 2 of a two-stage KB-grounded query rewriter.

ROLE:
Rewrite the user question into retrieval-ready queries.
Use ONLY dynamically discovered KB terms and metadata.
Do NOT answer the question.
Do NOT produce SQL.
Do NOT list companies.
Do NOT invent domain vocabulary.

KNOWLEDGE BASE METADATA:
{schema_ctx}

USER QUESTION:
{user_question}

STAGE 1 SEMANTIC PROBES:
These are semantic exploration probes only. They are NOT KB facts.
{probes_str}

DYNAMICALLY DISCOVERED KB TERMS:
You may use ONLY these KB terms as KB vocabulary.
{terms_str}

TERM SOURCES:
{sources_str}

EXPLICIT FILTERS:
{filters_str}

STRICT RULES:
1. final_rewritten_queries must use only:
   - words from the original user question,
   - column names / metadata wording,
   - explicit filter values,
   - dynamically discovered KB terms.
2. Do NOT use a Stage 1 probe term unless that term is also in the original question or dynamically discovered KB terms.
3. Do NOT invent new product names, company names, roles, tiers, materials, locations, or OEMs.
4. If discovered terms are weak, empty, or unrelated, set confidence="low" and keep final_rewritten_queries close to the original question.
5. If the question asks all/every/complete/full list/how many/count/total/highest/lowest/group/single point of failure, set requires_exhaustive_retrieval=true.
6. If "Georgia" is dataset scope, keep it in natural-language queries but do not force exact location match to the literal word Georgia.
7. negative_queries_or_terms_to_avoid should include obvious false-positive concepts only if supported by discovered terms or the question.

OUTPUT RULES:
Return a valid JSON object only.
No markdown.
No code fences.
No explanation.
No SQL.

EXPECTED JSON:
{{
  "stage": "kb_grounded_query_rewrite",
  "intent": "<list|count|compare|group|supplier_discovery|location|relationship|yes_no|other>",
  "explicit_filters": {{}},
  "target_columns": ["<schema_column_name>"],
  "mapped_user_phrases": [
    {{
      "user_phrase": "<phrase_from_question>",
      "kb_supported_terms": ["<term_from_discovered_terms>"],
      "mapping_source": "kb_discovered_terms",
      "confidence": "<high|medium|low>"
    }}
  ],
  "final_rewritten_queries": [
    "<query_1>",
    "<query_2>",
    "<query_3>"
  ],
  "negative_queries_or_terms_to_avoid": [],
  "requires_exhaustive_retrieval": false,
  "confidence": "<high|medium|low>",
  "warnings": []
}}"""


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str:
    payload: dict[str, Any] = {
        "model": config.QUERY_REWRITER_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(_cfg("QUERY_REWRITER_TEMPERATURE", 0.1)),
            "top_p": float(_cfg("QUERY_REWRITER_TOP_P", 0.85)),
            "num_predict": int(_cfg("QUERY_REWRITER_NUM_PREDICT", 1400)),
        },
    }

    # Optional; supported by modern Ollama, but kept configurable for compatibility.
    if bool(_cfg("QUERY_REWRITER_JSON_MODE", False)):
        payload["format"] = "json"

    num_ctx = _cfg("QUERY_REWRITER_NUM_CTX", None)
    if num_ctx:
        payload["options"]["num_ctx"] = int(num_ctx)

    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=config.QUERY_REWRITER_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()

    # DeepSeek-R1 style models may emit <think> blocks.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()

    # Remove common markdown fences if model ignored instructions.
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())

    return raw.strip()


# ── JSON parsing and validation ───────────────────────────────────────────────

def _extract_first_json_object(raw: str) -> dict | None:
    if not raw:
        return None

    text = raw.strip()
    start = text.find("{")
    if start < 0:
        return None

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except (json.JSONDecodeError, ValueError):
        # Fallback to greedy object extraction.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return None

    return obj if isinstance(obj, dict) else None


def _parse_and_validate_json(
    raw: str,
    required_keys: frozenset[str],
) -> dict | None:
    data = _extract_first_json_object(raw)
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
    retries = int(_cfg("MAX_REWRITER_RETRIES", 2))

    for attempt in range(max(1, retries)):
        try:
            raw = _call_ollama(current)
        except Exception:
            current = (
                prompt
                + "\n\nCRITICAL RETRY INSTRUCTION: Return ONLY one valid JSON object. No text."
            )
            continue

        result = _parse_and_validate_json(raw, required_keys)
        if result is not None:
            return result

        current = (
            prompt
            + "\n\nCRITICAL RETRY INSTRUCTION: Your previous output was invalid. "
              "Return ONLY one valid JSON object with all required keys. No text."
        )

    return None


# ── Stage output normalization / guardrails ───────────────────────────────────

def _normalise_intent(value: Any, user_question: str = "") -> str:
    if isinstance(value, list):
        value = next((v for v in value if str(v) in _VALID_INTENTS), "other")
    value = str(value).strip()

    if value not in _VALID_INTENTS:
        # Deterministic fallback from wording.
        q = user_question.lower()
        if _COUNT_RE.search(q):
            return "count"
        if _GROUP_RE.search(q):
            return "group"
        if any(w in q for w in ("supplier", "suppliers", "companies", "company")):
            return "supplier_discovery"
        return "other"

    return value


def _normalise_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return default


def _allowed_tokens_for_stage2(
    user_question: str,
    schema_index: dict[str, ColumnMeta],
    kb_terms: dict,
    explicit_filters: dict,
) -> set[str]:
    allowed = set(_GENERIC_QUERY_WORDS)
    allowed.update(_tokenize(user_question))

    for col in _schema_columns(schema_index):
        allowed.update(_tokenize(col))
        allowed.update(_tokenize(_column_description(col)))

    for term in _as_str_list(kb_terms.get("kb_discovered_terms", []), max_items=200):
        allowed.update(_tokenize(term))

    for val in (explicit_filters or {}).values():
        allowed.update(_tokenize(val))

    return {t for t in allowed if t}


def _unsupported_query_tokens(query: str, allowed_tokens: set[str]) -> list[str]:
    unsupported: list[str] = []
    for tok in _tokenize(query):
        if tok not in allowed_tokens:
            unsupported.append(tok)
    return _dedupe_preserve(unsupported)


def _filter_stage2_queries(
    queries: list[str],
    user_question: str,
    schema_index: dict[str, ColumnMeta],
    kb_terms: dict,
    explicit_filters: dict,
) -> tuple[list[str], list[str]]:
    allowed = _allowed_tokens_for_stage2(user_question, schema_index, kb_terms, explicit_filters)
    kept: list[str] = []
    warnings: list[str] = []

    for q in _as_str_list(queries, max_items=8):
        unsupported = _unsupported_query_tokens(q, allowed)
        # Allow very small noise, but reject if the model introduced multiple
        # unsupported content terms.
        if len(unsupported) >= int(_cfg("QUERY_REWRITER_MAX_UNSUPPORTED_TOKENS", 2)):
            warnings.append(f"dropped_query_unsupported_terms={unsupported}: {q}")
            continue
        kept.append(q)

    kept = _dedupe_preserve(kept)

    # Always keep the original user wording as a safe recall anchor.
    if user_question and user_question.lower() not in {q.lower() for q in kept}:
        kept.insert(0, user_question)

    return kept[: int(_cfg("QUERY_REWRITER_MAX_FINAL_QUERIES", 6))], warnings


def _normalise_mapped_user_phrases(
    mapped: Any,
    kb_terms: dict,
) -> tuple[list[dict[str, Any]], list[str]]:
    discovered = {t.lower(): t for t in _as_str_list(kb_terms.get("kb_discovered_terms", []), max_items=300)}
    out: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not isinstance(mapped, list):
        return [], ["mapped_user_phrases_not_list"]

    for item in mapped:
        if not isinstance(item, dict):
            continue

        phrase = str(item.get("user_phrase", "")).strip()
        raw_terms = _as_str_list(item.get("kb_supported_terms", []), max_items=30)
        supported_terms: list[str] = []

        for term in raw_terms:
            canonical = discovered.get(term.lower())
            if canonical:
                supported_terms.append(canonical)
            else:
                warnings.append(f"removed_unsupported_mapping_term={term}")

        conf = str(item.get("confidence", "low")).strip().lower()
        if conf not in _VALID_CONFIDENCE:
            conf = "low"
        if not supported_terms and raw_terms:
            conf = "low"

        out.append({
            "user_phrase": phrase,
            "kb_supported_terms": _dedupe_preserve(supported_terms),
            "mapping_source": "kb_discovered_terms",
            "confidence": conf,
        })

    return out, warnings


# ── Fallback builder ──────────────────────────────────────────────────────────

def build_fallback_stage2(
    user_question: str,
    stage1: dict,
    kb_terms: dict,
) -> dict:
    """
    Safe fallback when Stage 2 fails or KB evidence is sparse.
    Uses original question + Stage 1 probes.
    Preserves exhaustive intent.
    """
    probes = _as_str_list(stage1.get("semantic_probes", []), max_items=4)
    fallback_queries = _dedupe_preserve([user_question] + probes)
    discovered_count = len(_as_str_list(kb_terms.get("kb_discovered_terms", [])))

    requires_exhaustive = (
        _is_exhaustive_question(user_question)
        or str(stage1.get("intent", "")).lower() in {"count", "group"}
    )

    warnings = [
        "stage2_failed_or_weak_fallback_to_original_plus_stage1_probes",
        f"discovered_terms_count={discovered_count}",
    ]

    return {
        "stage": "kb_grounded_query_rewrite",
        "intent": _normalise_intent(stage1.get("intent", "other"), user_question),
        "explicit_filters": stage1.get("explicit_filters", {}),
        "target_columns": stage1.get("target_columns", []),
        "mapped_user_phrases": [],
        "final_rewritten_queries": fallback_queries,
        "negative_queries_or_terms_to_avoid": [],
        "requires_exhaustive_retrieval": requires_exhaustive,
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
    Score Stage 1 probe retrieval quality.

    This does not replace retrieval/reranking. It only decides whether the
    discovered vocabulary is strong enough to trust for Stage 2 rewriting.
    """
    if candidates_df is None or candidates_df.empty:
        return {
            "unique_row_count": 0,
            "avg_dense_similarity": 0.0,
            "avg_sparse_score": 0.0,
            "avg_fused_score": 0.0,
            "discovered_term_count": len(_as_str_list(kb_terms.get("kb_discovered_terms", []))),
            "explicit_filter_hits": 0,
            "explicit_filter_misses": len(explicit_filters or {}),
            "term_source_column_count": 0,
            "weak": True,
        }

    if "_row_id" in candidates_df.columns:
        unique_rows = int(candidates_df["_row_id"].nunique())
    else:
        unique_rows = int(len(candidates_df))

    def _avg_col(candidates: list[str]) -> float:
        for col in candidates:
            if col in candidates_df.columns:
                series = pd.to_numeric(candidates_df[col], errors="coerce")
                if series.notna().any():
                    return float(series.mean())
        return 0.0

    avg_dense = _avg_col(["_dense_score", "_score", "dense_score", "score"])
    avg_sparse = _avg_col(["_bm25_score", "_sparse_score", "bm25_score", "sparse_score"])
    avg_fused = _avg_col(["_rrf_score", "_fused_score", "rrf_score", "fused_score"])

    discovered_terms = len(_as_str_list(kb_terms.get("kb_discovered_terms", [])))

    term_source_cols: set[str] = set()
    for src in kb_terms.get("term_sources", []):
        if not isinstance(src, dict):
            continue
        term_source_cols.update(_as_str_list(src.get("source_columns", [])))

    filter_hits = 0
    filter_misses = 0

    if explicit_filters:
        for col, val in explicit_filters.items():
            # Metadata-only scope filters should not make retrieval weak.
            if str(col).startswith("__"):
                filter_hits += 1
                continue

            # "Georgia" is often dataset scope in this project; do not require
            # exact literal Georgia in city/county rows.
            if _looks_like_georgia_scope_filter(val):
                filter_hits += 1
                continue

            if col in candidates_df.columns:
                pattern = re.escape(str(val).lower())
                hit = candidates_df[col].astype(str).str.lower().str.contains(
                    pattern, na=False, regex=True
                ).any()
                if hit:
                    filter_hits += 1
                else:
                    filter_misses += 1
            else:
                filter_misses += 1

    min_rows = int(_cfg("PROBE_MIN_ROWS", 5))
    min_terms = int(_cfg("KB_TERM_MIN_DISCOVERED", 2))
    min_source_cols = int(_cfg("KB_TERM_MIN_SOURCE_COLUMNS", 1))

    weak = (
        unique_rows < min_rows
        or discovered_terms < min_terms
        or len(term_source_cols) < min_source_cols
        or filter_misses > 0
    )

    return {
        "unique_row_count": unique_rows,
        "avg_dense_similarity": round(avg_dense, 4),
        "avg_sparse_score": round(avg_sparse, 4),
        "avg_fused_score": round(avg_fused, 4),
        "discovered_term_count": discovered_terms,
        "explicit_filter_hits": filter_hits,
        "explicit_filter_misses": filter_misses,
        "term_source_column_count": len(term_source_cols),
        "weak": bool(weak),
    }


# ── Public two-stage API ──────────────────────────────────────────────────────

def stage1_probe_generation(
    user_question: str,
    schema_index: dict[str, ColumnMeta],
) -> dict | None:
    """
    Stage 1: Generate semantic probes from user question + metadata only.
    Returns validated/normalized dict or None on failure.
    """
    if not bool(_cfg("QUERY_REWRITER_ENABLED", True)):
        return None

    schema_ctx = _build_stage1_schema_context(schema_index)
    prompt = _build_stage1_prompt(user_question, schema_ctx)
    result = _call_with_retry(prompt, _STAGE1_REQUIRED)

    if result is None:
        return None

    intent = _normalise_intent(result.get("intent"), user_question)

    probes = _as_str_list(result.get("semantic_probes", []), max_items=10)
    # Always include original question as a recall anchor.
    probes = _dedupe_preserve([user_question] + probes)
    if not probes:
        return None

    target_columns = _normalise_target_columns(result.get("target_columns", []), schema_index)

    explicit_filters = _normalise_explicit_filters(
        result.get("explicit_filters", {}),
        schema_index,
    )

    requires_broad = (
        _normalise_bool(result.get("requires_broad_retrieval"), default=True)
        or _is_exhaustive_question(user_question)
        or intent in {"count", "group", "supplier_discovery", "list"}
    )

    return {
        "intent": intent,
        "target_columns": target_columns,
        "semantic_probes": probes[:10],
        "explicit_filters": explicit_filters,
        "requires_broad_retrieval": bool(requires_broad),
        "stage": "semantic_probe_generation",
        "status": "semantic_only_not_kb_verified",
    }


def stage2_kb_grounded_rewrite(
    user_question: str,
    schema_index: dict[str, ColumnMeta],
    stage1: dict,
    kb_terms: dict,
    explicit_filters: dict,
) -> dict | None:
    """
    Stage 2: Rewrite user question using dynamically discovered KB terms only.
    Returns validated/normalized dict or None on failure.
    """
    if not bool(_cfg("QUERY_REWRITER_ENABLED", True)):
        return None

    # Critical fix: Stage 2 uses metadata only, not schema sample values.
    schema_ctx = _build_metadata_schema_context(schema_index)

    probes = _as_str_list(stage1.get("semantic_probes", []), max_items=10)
    safe_filters = _normalise_explicit_filters(
        explicit_filters or stage1.get("explicit_filters", {}),
        schema_index,
    )

    prompt = _build_stage2_prompt(
        user_question=user_question,
        schema_ctx=schema_ctx,
        semantic_probes=probes,
        kb_terms=kb_terms or {},
        explicit_filters=safe_filters,
    )

    result = _call_with_retry(prompt, _STAGE2_REQUIRED)
    if result is None:
        return None

    intent = _normalise_intent(result.get("intent", stage1.get("intent", "other")), user_question)
    target_columns = _normalise_target_columns(result.get("target_columns", []), schema_index)
    if not target_columns:
        target_columns = _normalise_target_columns(stage1.get("target_columns", []), schema_index)

    rewritten_queries, query_warnings = _filter_stage2_queries(
        queries=_as_str_list(result.get("final_rewritten_queries", []), max_items=8),
        user_question=user_question,
        schema_index=schema_index,
        kb_terms=kb_terms or {},
        explicit_filters=safe_filters,
    )

    if not rewritten_queries:
        return None

    mapped_phrases, mapping_warnings = _normalise_mapped_user_phrases(
        result.get("mapped_user_phrases", []),
        kb_terms or {},
    )

    confidence = str(result.get("confidence", "low")).strip().lower()
    if confidence not in _VALID_CONFIDENCE:
        confidence = "low"

    warnings = _as_str_list(result.get("warnings", []), max_items=30)
    warnings.extend(query_warnings)
    warnings.extend(mapping_warnings)
    warnings = _dedupe_preserve(warnings)

    # If we dropped unsupported terms or have no discovered terms, lower confidence.
    if query_warnings or len(_as_str_list((kb_terms or {}).get("kb_discovered_terms", []))) == 0:
        confidence = "low"

    requires_exhaustive = (
        _normalise_bool(result.get("requires_exhaustive_retrieval"), default=False)
        or _is_exhaustive_question(user_question)
        or intent in {"count", "group"}
    )

    negative_terms = _as_str_list(result.get("negative_queries_or_terms_to_avoid", []), max_items=20)

    return {
        "stage": "kb_grounded_query_rewrite",
        "intent": intent,
        "explicit_filters": safe_filters,
        "target_columns": target_columns,
        "mapped_user_phrases": mapped_phrases,
        "final_rewritten_queries": rewritten_queries,
        "negative_queries_or_terms_to_avoid": negative_terms,
        "requires_exhaustive_retrieval": bool(requires_exhaustive),
        "confidence": confidence,
        "warnings": warnings,
    }


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
    """
    Legacy single-pass rewriter.

    Kept for backward compatibility. The main two-stage pipeline should call
    stage1_probe_generation() and stage2_kb_grounded_rewrite() instead.
    """
    if not unmatched_words or not bool(_cfg("QUERY_REWRITER_ENABLED", True)):
        return question

    schema_ctx = _build_schema_context(schema_index)
    unmatched_str = ", ".join(f'"{w}"' for w in unmatched_words)
    prompt = f"""\
You are a query rewriting assistant for a Georgia EV supply chain knowledge base.

TASK:
Rewrite the user's question by replacing ONLY the ambiguous terms listed below
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
  2. Use EXACT values from the schema when rewriting.
  3. If a term maps to multiple KB values, join with " or ".
  4. Leave terms unchanged if you cannot confidently map them.
  5. Output ONLY the rewritten question. No explanation, no quotes.

REWRITTEN QUESTION:"""

    try:
        rewritten = _call_ollama(prompt)
    except Exception:
        return question

    return rewritten if _is_plausible(question, rewritten) else question