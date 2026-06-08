"""Deterministic answer formatting, with an optional Ollama-grounded path.

Per the execution README, answers start as simple deterministic templates over
the retrieved evidence. When ``use_llm`` is requested, the same evidence is
handed to the local Ollama model (reusing ``runtime_pipeline.generation``) to
produce a grounded natural-language answer. The LLM never writes SQL and never
sees the database directly — only the already-retrieved evidence.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MAX_LISTED_ROWS = 25
_CHUNK_PREVIEW_CHARS = 300


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

def llm_answer(question: str, evidence: dict[str, Any], deterministic: str) -> str:
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
        prompt = _build_prompt(question, deterministic)
        answer = adapter.generate(prompt)
        return answer or deterministic
    except Exception as exc:  # never let LLM issues break the run
        logger.warning("LLM answer generation failed, using deterministic: %s", exc)
        return deterministic


def _build_prompt(question: str, evidence_text: str) -> str:
    return (
        "You are a careful analyst for a Georgia EV supply-chain knowledge base.\n"
        "Answer the user's question using ONLY the evidence below. Do not invent "
        "companies, numbers, or facts that are not present. If the evidence is "
        "insufficient, say so plainly.\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Answer:"
    )
