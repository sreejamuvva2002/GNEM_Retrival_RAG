"""
phase4_agent/text_to_sql.py
==============================================================
Text-to-SQL: LLM generates SQL directly from the question + a runtime-built
schema block.

WHY THIS REPLACES entity_extractor + sql_retriever rules for some queries:
  Rules break for edge cases:
    "Tier 1 only"       → needs exact match, not ilike
    "at least 500 emp"  → needs >= filter
    "top 5 by county"   → needs LIMIT 5
    "excluding OEMs"    → needs NOT LIKE filter

  The LLM already knows SQL. It reads the question AND the schema block
  and generates the correct SQL for any phrasing — no rules needed.

SCHEMA PROVIDED TO LLM:
  Built at runtime from CANONICAL_FIELDS + FIELD_TYPES + a small sample of
  distinct values per column from metadata_loader. NOTHING in source code
  hardcodes a real company / county / OEM / tier / role / facility value.
  All example values shown to the LLM are sampled from the live KB.

  Few-shot Q&A pairs that previously appeared in this file have been
  removed: every pair encoded a domain-specific question wording and the
  expected SQL, which was the most direct form of golden-question shape
  leakage in the codebase.

SAFETY:
  - Read-only: only SELECT queries allowed (validated before execution)
  - Schema-locked: LLM only sees gev_companies columns
  - Result-capped: LIMIT 200 max
"""
from __future__ import annotations

import re

import httpx

from filters_and_validation.metadata_loader import loader as kb_loader
from shared.config import Config
from shared.db import get_session
from shared.logger import get_logger
from shared.metadata_schema import (
    CANONICAL_FIELDS,
    FIELD_TYPES,
    SUPPORTED_OPERATORS,
)

logger = get_logger("phase4.text_to_sql")


def _build_schema_block(per_field: int = 5) -> str:
    """
    Compose the schema description shown to the LLM.

    Column names come from CANONICAL_FIELDS (approved hardcoding). Example
    values for each column are sampled from the live KB via metadata_loader
    so the prompt always reflects current data and never carries a literal
    company/county/OEM string in source code.
    """
    samples = kb_loader.sample_distinct_values(per_field=per_field)
    lines: list[str] = ["TABLE: gev_companies", "COLUMNS:"]
    lines.append("  id                      INTEGER  (primary key)")
    for col in CANONICAL_FIELDS.values():
        ftype = FIELD_TYPES.get(col, "text").upper()
        examples = samples.get(col, [])
        ex_str = ", ".join(repr(v) for v in examples[:per_field]) if examples else ""
        ex_clause = f"  e.g. {ex_str}" if ex_str else ""
        lines.append(f"  {col:<25} {ftype:<8}{ex_clause}")
    lines.append("")
    lines.append(f"OPERATORS (use only these): {', '.join(sorted(SUPPORTED_OPERATORS))}")
    lines.append("")
    lines.append("RULES:")
    lines.append("  - Use exact match (=) when the question says 'only', 'strictly', 'exactly'.")
    lines.append("  - Use ILIKE '%value%' when the question says 'including', 'or higher', 'broad'.")
    lines.append("  - Always add ORDER BY for ranked / top-N questions.")
    lines.append("  - Always add LIMIT (default 50, max 200).")
    lines.append("  - Do NOT invent column names or values that are not present above.")
    return "\n".join(lines)


_SQL_PROMPT = """You are a PostgreSQL expert. Generate a single SQL SELECT query for the question below.

DATABASE SCHEMA:
{schema}

OUTPUT RULES:
1. Use ONLY the gev_companies table.
2. Use ONLY the columns and operators listed in the schema above.
3. Output ONLY the SQL query — no explanation, no markdown, no backticks.

QUESTION: {question}
SQL:"""


def generate_sql(question: str) -> str:
    """
    Use the LLM to generate a SQL query from the question.

    Few-shot examples are no longer hardcoded in source. If a Phase-5
    few-shot store is configured, KB-grounded examples are injected at
    runtime; otherwise the LLM reasons from the schema block alone.
    """
    cfg = Config.get()
    schema_block = _build_schema_block()

    fewshot_block = ""
    try:
        from retrievals.few_shot_retriever import get_few_shot_block
        block = get_few_shot_block(question, query_type="sql", top_k=3)
        if block:
            fewshot_block = "\n\nEXAMPLES (retrieved from approved store):\n" + block
            logger.info("Phase 5: injected dynamic SQL examples")
    except Exception as exc:
        logger.debug("Phase 5 few-shot unavailable (%s) — proceeding with schema-only prompt", exc)

    prompt = _SQL_PROMPT.format(
        schema=schema_block + fewshot_block,
        question=question,
    )

    payload = {
        "model":  cfg.ollama_cypher_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 300,
            "num_ctx":     4096,
        },
    }
    try:
        url = f"{cfg.ollama_base_url}/api/generate"
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            raw = str(resp.json().get("response", "")).strip()
            sql = _clean_sql(raw)
            logger.info("Generated SQL (%d chars): %s", len(sql), sql[:120])
            return sql
    except Exception as exc:
        logger.error("SQL generation failed: %s", exc)
        return ""


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and extract the SELECT statement."""
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip("`").strip()
    match = re.search(r"(SELECT\b.*?)(?:;|$)", raw, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip() + ";"
    return raw.strip()


def _is_safe_sql(sql: str) -> bool:
    """
    Safety check: only allow SELECT queries.
    Rejects INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, etc.
    """
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        logger.warning("Rejected non-SELECT SQL: %s", sql[:80])
        return False
    dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE",
                 "ALTER", "CREATE", "GRANT", "REVOKE", "EXEC"]
    for kw in dangerous:
        if re.search(rf"\b{kw}\b", sql_upper):
            logger.warning("Rejected SQL with dangerous keyword '%s': %s", kw, sql[:80])
            return False
    return True


def execute_sql_safe(sql: str) -> list[dict]:
    """
    Execute a generated SQL query safely.
    Returns list of row dicts, or empty list on error.
    """
    if not sql or not _is_safe_sql(sql):
        return []

    if "LIMIT" not in sql.upper():
        sql = sql.rstrip(";") + " LIMIT 200;"

    session = get_session()
    try:
        from sqlalchemy import text
        result = session.execute(text(sql))
        keys = list(result.keys())
        rows = [dict(zip(keys, row)) for row in result.fetchall()]
        logger.info("SQL executed: %d rows returned", len(rows))
        return rows
    except Exception as exc:
        logger.error("SQL execution error: %s | SQL: %s", exc, sql[:200])
        return []
    finally:
        session.close()


def text_to_sql(question: str) -> list[dict]:
    """
    Full pipeline: question → generate SQL → execute → return rows.
    Returns empty list if generation or execution fails.
    """
    sql = generate_sql(question)
    if not sql:
        return []
    return execute_sql_safe(sql)
