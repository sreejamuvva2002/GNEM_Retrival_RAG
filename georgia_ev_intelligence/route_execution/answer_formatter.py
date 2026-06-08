"""Deterministic answer formatting, with an optional Ollama-grounded path.

Per the execution README, answers start as simple deterministic templates over
the retrieved evidence. When ``use_llm`` is requested, the same evidence is
handed to the local Ollama model (reusing ``runtime_pipeline.generation``) to
produce a grounded natural-language answer. The LLM never writes SQL and never
sees the database directly — only the already-retrieved evidence.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_MAX_LISTED_ROWS = 25
_CHUNK_PREVIEW_CHARS = 300

_ROUTE_CONTEXT_FIELDS = (
    "route",
    "operation",
    "entities",
    "context_entities",
    "raw_filters",
    "resolved_filters",
    "search_filters",
    "requested_columns",
    "group_by",
    "sort_by",
    "limit",
    "query_focus",
    "retrieval_sources",
)

_EVIDENCE_FIELDS_BY_TYPE = {
    "structured_rows": ("columns", "rows"),
    "count": ("count",),
    "group_counts": ("groups",),
    "group_aggregates": ("groups", "group_by", "aggregate_column"),
    "document_chunks": ("chunks", "parents", "fallback"),
    "geo_results": ("center", "radius_miles", "rows"),
    "ranked_alternatives": ("disrupted_company", "alternatives"),
    "clarification": ("reason", "query", "clarification"),
}


def format_structured_rows(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render structured rows as a numbered, human-readable list."""
    if not rows:
        return "Found 0 matching records."

    lines = [f"Found {len(rows)} matching records."]
    for idx, row in enumerate(rows[:_MAX_LISTED_ROWS], start=1):
        company = row.get("company") or "(unknown company)"
        lines.append(f"\n{idx}. {company}")
        for col in columns:
            if col == "company":
                continue
            value = row.get(col)
            if value in (None, ""):
                continue
            lines.append(f"   {_humanize(col)}: {value}")
    if len(rows) > _MAX_LISTED_ROWS:
        lines.append(f"\n… and {len(rows) - _MAX_LISTED_ROWS} more.")
    return "\n".join(lines)


def format_count(count: int) -> str:
    return f"Found {count} matching records."


def format_group_counts(rows: list[dict[str, Any]], group_by: list[str]) -> str:
    """Render grouped counts (``GROUP BY`` results with a count column)."""
    if not rows:
        return "No groups matched."
    label = ", ".join(_humanize(g) for g in group_by) or "group"
    lines = [f"Counts by {label}:"]
    for row in rows:
        key = " / ".join(str(row.get(g, "")) for g in group_by)
        count = row.get("count", "")
        lines.append(f"- {key or '(blank)'}: {count}")
    return "\n".join(lines)


def format_group_aggregates(
    rows: list[dict[str, Any]],
    group_by: list[str],
    aggregate_column: str,
) -> str:
    """Render grouped numeric aggregate results."""
    if not rows:
        return "No groups matched."
    label = ", ".join(_humanize(g) for g in group_by) or "group"
    metric = _humanize(aggregate_column)
    lines = [f"{metric} by {label}:"]
    for row in rows:
        key = " / ".join(str(row.get(g, "")) for g in group_by)
        lines.append(f"- {key or '(blank)'}: {row.get(aggregate_column, '')}")
    return "\n".join(lines)


def format_document_chunks(previews: list[dict[str, Any]]) -> str:
    """Render document evidence (parent chunk previews)."""
    if not previews:
        return "No supporting evidence found."
    lines = ["I found relevant supporting evidence:"]
    for item in previews:
        text = (item.get("text") or "").strip().replace("\n", " ")
        if len(text) > _CHUNK_PREVIEW_CHARS:
            text = text[:_CHUNK_PREVIEW_CHARS].rstrip() + "…"
        lines.append(f"- {text}")
    return "\n".join(lines)


def _humanize(column: str) -> str:
    return column.replace("_", " ").strip().title()


# ---------------------------------------------------------------------------
# Optional LLM-grounded answer
# ---------------------------------------------------------------------------

def llm_answer(
    question: str,
    evidence: dict[str, Any],
    deterministic: str,
    *,
    final_route: dict[str, Any] | None = None,
) -> str:
    """Generate a grounded answer from evidence using the local Ollama model.

    Falls back to the deterministic answer if the model is unavailable so a
    single flaky call never aborts a batch run.
    """
    try:
        from georgia_ev_intelligence.runtime_pipeline.generation.llm_adapter import (
            OllamaAdapter,
        )
        from georgia_ev_intelligence.shared import config

        adapter = OllamaAdapter(config.OLLAMA_LLM_MODEL)
        prompt = _build_prompt(
            question,
            route_context=_route_context(final_route or {}),
            grounded_evidence=_grounded_evidence(evidence),
            deterministic_answer=deterministic,
        )
        answer = adapter.generate(prompt)
        return answer or deterministic
    except Exception as exc:  # never let LLM issues break the run
        logger.warning("LLM answer generation failed, using deterministic: %s", exc)
        return deterministic


def _route_context(final_route: dict[str, Any]) -> dict[str, Any]:
    """Return the validated routing/filter fields useful for answer grounding."""
    return {
        field: final_route[field]
        for field in _ROUTE_CONTEXT_FIELDS
        if field in final_route and final_route[field] not in (None, [], {}, "")
    }


# Row keys fetched only so the UI can place a company on the map; never useful
# to the answer LLM and a frequent source of spurious "(lat, lon)" output.
_NON_GROUNDING_ROW_KEYS = ("latitude", "longitude")


def _strip_map_keys(rows: Any) -> Any:
    """Drop map-only coordinate keys from evidence rows before grounding."""
    if not isinstance(rows, (list, tuple)):
        return rows
    cleaned = []
    for row in rows:
        if isinstance(row, dict):
            cleaned.append({k: v for k, v in row.items() if k not in _NON_GROUNDING_ROW_KEYS})
        else:
            cleaned.append(row)
    return cleaned


def _grounded_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Whitelist factual evidence for the answer LLM, excluding SQL/debug data."""
    if not isinstance(evidence, dict):
        return {}

    evidence_type = str(evidence.get("type") or "")
    grounded: dict[str, Any] = {"type": evidence_type}
    for field in _EVIDENCE_FIELDS_BY_TYPE.get(evidence_type, ()):
        if field in evidence:
            grounded[field] = _strip_map_keys(evidence[field]) if field == "rows" else evidence[field]
    return grounded


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=str)


def _build_prompt(
    question: str,
    *,
    route_context: dict[str, Any],
    grounded_evidence: dict[str, Any],
    deterministic_answer: str,
) -> str:
    return (
        "You are a careful analyst for a Georgia EV supply-chain knowledge base.\n"
        "Answer the user's question using ONLY the retrieved evidence JSON below. "
        "The validated route JSON defines the requested scope, filters, grouping, "
        "and output columns; it is not itself factual evidence. Do not invent "
        "companies, counts, roles, products, locations, or other facts.\n"
        "For structured and geo rows, every returned row already satisfies the "
        "validated filters. Include all returned rows, preserve their values, and "
        "never exclude or re-filter a row based on your own interpretation. "
        "For document retrieval, ground the answer in the full parent-context text. "
        "Do not mention internal route names, SQL, JSON, or retrieval mechanics. "
        "If the evidence is insufficient, say so plainly.\n\n"
        f"Question:\n{question}\n\n"
        f"Validated route and filters JSON:\n{_json_text(route_context)}\n\n"
        f"Retrieved evidence JSON:\n{_json_text(grounded_evidence)}\n\n"
        "Deterministic fallback answer:\n"
        f"{deterministic_answer}\n\n"
        "Answer:"
    )
