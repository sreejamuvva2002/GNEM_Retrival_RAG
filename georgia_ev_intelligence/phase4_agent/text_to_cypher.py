"""
phase4_agent/text_to_cypher.py
==============================================================
Text-to-Cypher: converts a natural-language question into Cypher using the
local Gemma model, then executes it against Neo4j with one self-healing
retry on failure.

WHY GEMMA FOR CYPHER GENERATION:
  - Faster than the main answer model (~1-3s vs 5-10s)
  - Strong code understanding from Google training
  - Smaller context needed (just schema + question)
  - The main answer model is reserved for answer synthesis

SCHEMA PROVIDED TO LLM:
  Built at runtime from GRAPH_LABELS + GRAPH_RELATIONSHIPS plus a small
  sample of distinct property values per Company column from the live KB.
  NOTHING in source code hardcodes a real company / OEM / county / tier /
  role / facility value. All example values shown to the LLM are sampled
  from gev_companies via metadata_loader.

  The block of question-to-Cypher few-shot pairs that previously appeared
  here has been removed. Each pair encoded a domain-specific question
  wording (specific OEM brands, tier values, role values, and county
  names) and was the most direct form of golden-question shape leakage
  in the codebase. The LLM now reasons from the schema block alone;
  KB-grounded examples may be injected at runtime via the Phase 5
  few-shot store if configured.

SCHEMA NOTE:
  Neo4j Company nodes use c.name (NOT c.company_name).
  All queries must alias: c.name AS company_name.
"""
from __future__ import annotations

import re
import httpx

from phase3_graph.graph_loader import get_driver
from phase4_agent.metadata_loader import loader as kb_loader
from shared.config import Config
from shared.logger import get_logger
from shared.metadata_schema import (
    CANONICAL_FIELDS,
    FIELD_TYPES,
    GRAPH_LABELS,
    GRAPH_RELATIONSHIPS,
)

logger = get_logger("phase4.text_to_cypher")


# Mapping from gev_companies column name to the Company node property name.
# 'company_name' is exposed as `c.name` on the Company node; everything else
# uses the column name directly. Defined here, not derived from a magic dict
# elsewhere, because the Neo4j loader is the source of truth and this name
# mismatch is intentional.
_COLUMN_TO_NODE_PROPERTY: dict[str, str] = {
    "company_name": "name",
}


def _node_property(column: str) -> str:
    return _COLUMN_TO_NODE_PROPERTY.get(column, column)


def _build_schema_block(per_field: int = 5) -> str:
    """Compose the Neo4j schema description with KB-sampled example values."""
    samples = kb_loader.sample_distinct_values(per_field=per_field)
    lines: list[str] = ["GRAPH SCHEMA — Georgia EV Supply Chain", "", "NODE LABELS:"]
    # Company carries every domain attribute; list other labels as bare names.
    lines.append("  (:Company)")
    for col in CANONICAL_FIELDS.values():
        prop = _node_property(col)
        ftype = FIELD_TYPES.get(col, "text").upper()
        examples = samples.get(col, [])
        ex_str = ", ".join(repr(v) for v in examples[:per_field]) if examples else ""
        ex_clause = f"  e.g. {ex_str}" if ex_str else ""
        lines.append(f"    c.{prop:<23} {ftype:<8}{ex_clause}")
    lines.append("")
    other_labels = [lbl for lbl in sorted(GRAPH_LABELS) if lbl != "Company"]
    for lbl in other_labels:
        lines.append(f"  (:{lbl})  name STRING")
    lines.append("")
    lines.append("RELATIONSHIPS:")
    for rel in sorted(GRAPH_RELATIONSHIPS):
        lines.append(f"  (Company)-[:{rel}]->(...)")
    lines.append("")
    lines.append("RULES:")
    lines.append("  - Company node property is `name` (NOT `company_name`).")
    lines.append("  - Always alias `c.name AS company_name` in the RETURN clause.")
    lines.append("  - Use toLower() + CONTAINS for case-insensitive partial matching.")
    lines.append("  - Use IS NOT NULL checks before CONTAINS on nullable fields.")
    lines.append("  - Add LIMIT 50 to prevent huge result sets.")
    lines.append("  - Do NOT invent property names or label names not listed above.")
    return "\n".join(lines)


_CYPHER_PROMPT = """You are a Neo4j Cypher expert for the Georgia EV Supply Chain Intelligence System.

GRAPH SCHEMA:
{schema}

OUTPUT RULES:
1. Return ONLY the raw Cypher query — no explanation, no markdown, no backticks.
2. ALWAYS alias `c.name AS company_name` in the RETURN clause.
3. Use toLower() + CONTAINS for case-insensitive partial matching.
4. Always add IS NOT NULL check before CONTAINS on nullable fields.
5. Use LIMIT 50 to prevent huge result sets.
6. NEVER use `c.company_name` — the Company node property is `name`.

QUESTION: {question}

CYPHER:"""


def generate_cypher(question: str, error_feedback: str = "") -> str:
    """
    Use Gemma (fast, code-focused) to generate a Cypher query.
    If error_feedback is provided, includes it for self-correction.
    """
    cfg = Config.get()
    schema_block = _build_schema_block()

    fewshot_block = ""
    try:
        from phase5_fewshot.few_shot_retriever import get_few_shot_block
        block = get_few_shot_block(question, query_type="cypher", top_k=3)
        if block:
            fewshot_block = "\n\nEXAMPLES (retrieved from approved store):\n" + block
            logger.info("Phase 5: injected dynamic Cypher examples")
    except Exception as exc:
        logger.debug("Phase 5 few-shot unavailable (%s) — proceeding with schema-only prompt", exc)

    prompt_question = question
    if error_feedback:
        prompt_question = (
            f"{question}\n\n"
            f"PREVIOUS ATTEMPT FAILED WITH ERROR:\n{error_feedback}\n"
            f"Fix the Cypher query to avoid this error."
        )

    prompt = _CYPHER_PROMPT.format(
        schema=schema_block + fewshot_block,
        question=prompt_question,
    )

    payload = {
        "model": cfg.ollama_cypher_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 400,
            "num_ctx": 4096,
        },
    }

    try:
        url = f"{cfg.ollama_base_url}/api/generate"
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            raw = str(resp.json().get("response", "")).strip()
            cypher = _clean_cypher(raw)
            logger.info("Generated Cypher (%d chars): %s", len(cypher), cypher[:100])
            return cypher
    except Exception as exc:
        logger.error("Cypher generation failed: %s", exc)
        return ""


def _clean_cypher(raw: str) -> str:
    """Remove markdown fences, leading/trailing whitespace, thinking tags."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```(?:cypher)?", "", raw, flags=re.IGNORECASE).strip("`").strip()
    match = re.search(r"\b(MATCH|WITH|RETURN|CALL)\b", raw, re.IGNORECASE)
    if match:
        raw = raw[match.start():]
    return raw.strip()


def execute_cypher(cypher: str) -> list[dict]:
    """Execute Cypher query against Neo4j. Returns list of row dicts."""
    if not cypher:
        return []
    driver = get_driver()
    try:
        with driver.session() as s:
            result = s.run(cypher)
            rows = result.data()
            logger.info("Cypher executed: %d rows returned", len(rows))
            return rows
    except Exception as exc:
        err_str = str(exc)
        logger.warning("Cypher execution error: %s", err_str)
        if any(kw in err_str.lower() for kw in ("routing", "connection", "no data", "defunct")):
            logger.info("Neo4j connection lost — resetting driver for reconnect")
            from phase3_graph.graph_loader import close_driver
            try:
                close_driver()
            except Exception:
                pass
        raise


def execute_cypher_safe(question: str) -> list[dict]:
    """
    Generate Cypher and execute with one self-healing retry.
    If generation fails → return []
    If execution fails → regenerate with error message → retry once
    If retry also fails → return []
    """
    cypher = generate_cypher(question)
    if not cypher:
        logger.warning("Cypher generation produced empty string")
        return []

    error_msg = ""
    try:
        return execute_cypher(cypher)
    except Exception as exc:
        error_msg = str(exc)
        logger.warning("Cypher attempt 1 failed, self-healing: %s", error_msg)

    cypher2 = generate_cypher(question, error_feedback=error_msg)
    if not cypher2:
        return []
    try:
        return execute_cypher(cypher2)
    except Exception as exc2:
        logger.error("Cypher attempt 2 also failed: %s", exc2)
        return []


def normalize_cypher_results(rows: list[dict]) -> list[dict]:
    """
    Normalize Neo4j result rows to the standard company-row format so they
    feed the existing formatter pipeline.
    """
    normalized = []
    for row in rows:
        name = (
            row.get("company_name")
            or row.get("c.name")
            or row.get("name")
            or ""
        )
        normalized.append({
            "company_name":          str(name),
            "tier":                  str(row.get("tier") or row.get("c.tier") or ""),
            "ev_supply_chain_role":  str(row.get("ev_supply_chain_role") or row.get("c.ev_supply_chain_role") or ""),
            "ev_battery_relevant":   str(row.get("ev_battery_relevant") or row.get("c.ev_battery_relevant") or ""),
            "location_county":       str(row.get("location_county") or row.get("c.location_county") or ""),
            "employment":            row.get("employment") or row.get("c.employment"),
            "products_services":     str(row.get("products_services") or row.get("c.products_services") or ""),
            "industry_group":        str(row.get("industry_group") or row.get("c.industry_group") or ""),
            "facility_type":         str(row.get("facility_type") or row.get("c.facility_type") or ""),
            "primary_oems":          str(row.get("primary_oems") or row.get("oem_name") or ""),
        })
    return normalized
