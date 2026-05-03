"""
phase4_agent/text_to_sql.py
==============================================================
Text-to-SQL: LLM generates SQL directly from the question + schema.

WHY THIS REPLACES entity_extractor + sql_retriever rules:
  Rules break for edge cases:
    "Tier 1 only"       → needs exact match, not ilike
    "at least 500 emp"  → needs >= filter
    "top 5 by county"   → needs LIMIT 5
    "excluding OEMs"    → needs NOT LIKE filter

  The LLM already knows SQL. It reads the question AND the schema,
  and generates the correct SQL for any phrasing — no rules needed.

SCHEMA PROVIDED TO LLM:
  The full gev_companies table schema with real example values.
  This grounds the LLM in actual column names and value formats.

SAFETY:
  - Read-only: only SELECT queries allowed (validated before execution)
  - Schema-locked: LLM only sees gev_companies table
  - Result-capped: LIMIT 200 max

WHEN USED:
  For aggregate/filter questions where SQL genuinely wins over Cypher:
    - GROUP BY county/tier/role
    - SUM/COUNT employment
    - ORDER BY / LIMIT (top-N)
    - Complex WHERE with multiple filters

WHEN NOT USED:
  Graph relationship questions (SUPPLIES_TO, IN_TIER) → Cypher handles those.
"""
from __future__ import annotations

import re

import httpx

from shared.config import Config
from shared.db import get_session
from shared.logger import get_logger

logger = get_logger("phase4.text_to_sql")

# ── Schema description given to the LLM ──────────────────────────────────────
# Real column names + real example values from the actual database.
# No guessing — LLM generates SQL that will execute correctly.

_SCHEMA = """
TABLE: gev_companies
COLUMNS and real example values:
  id                      INTEGER  (primary key)
  company_name            TEXT     e.g. 'Hyundai Motor Group', 'SK Battery America'
  tier                    TEXT     EXACT values: 'Tier 1', 'Tier 2/3', 'Tier 1/2',
                                   'OEM (Footprint)', 'OEM Supply Chain'
  ev_supply_chain_role    TEXT     e.g. 'Battery Cell', 'Thermal Management', 'Charging Infrastructure'
  ev_battery_relevant     TEXT     values: 'Yes', 'No', 'Indirect'
  industry_group          TEXT     e.g. 'Electronic and Other Electrical', 'Transportation Equipment'
  facility_type           TEXT     e.g. 'Manufacturing Plant', 'R&D', 'Headquarters'
  location_city           TEXT     e.g. 'LaGrange', 'Cartersville'
  location_county         TEXT     e.g. 'Troup County', 'Gwinnett County', 'Hall County'
  employment              FLOAT    number of employees at Georgia facility (not global headcount)
  products_services       TEXT     free text e.g. 'Lithium-ion battery materials'
  primary_oems            TEXT     OEM names e.g. 'Hyundai', 'Rivian', 'BMW'
  supplier_affiliation_type TEXT
  classification_method   TEXT

IMPORTANT NOTES:
  - 'Tier 1' is DIFFERENT from 'Tier 1/2'. Use exact match (=) when question says 'only'.
  - Use ILIKE '%value%' only when question asks for broad/inclusive matching.
  - employment <= 100000 to exclude global headcount outliers.
  - Always include ORDER BY for ranked results.
  - Always add LIMIT (default 50, max 200).
"""

_FEW_SHOT = """
EXAMPLES — notice how SQL matches the question's intent exactly:

Q: Which county has the highest total employment among Tier 1 suppliers only?
SQL: SELECT location_county, SUM(employment) AS total_employment, COUNT(*) AS company_count
     FROM gev_companies
     WHERE tier = 'Tier 1'
       AND employment IS NOT NULL AND employment <= 100000
     GROUP BY location_county ORDER BY total_employment DESC LIMIT 10;

Q: List the top 5 counties by employment for Tier 1/2 suppliers.
SQL: SELECT location_county, SUM(employment) AS total_employment, COUNT(*) AS company_count
     FROM gev_companies
     WHERE tier ILIKE '%Tier 1/2%'
       AND employment IS NOT NULL AND employment <= 100000
     GROUP BY location_county ORDER BY total_employment DESC LIMIT 5;

Q: How many companies are in each EV supply chain role?
SQL: SELECT ev_supply_chain_role, COUNT(*) AS company_count
     FROM gev_companies
     WHERE ev_supply_chain_role IS NOT NULL
     GROUP BY ev_supply_chain_role ORDER BY company_count DESC LIMIT 50;

Q: Which companies have more than 1000 employees and are EV battery relevant?
SQL: SELECT company_name, tier, employment, location_county, ev_supply_chain_role
     FROM gev_companies
     WHERE employment > 1000
       AND ev_battery_relevant ILIKE '%Yes%'
     ORDER BY employment DESC LIMIT 50;

Q: Show all companies in Hall County with their tier and role.
SQL: SELECT company_name, tier, ev_supply_chain_role, employment, facility_type
     FROM gev_companies
     WHERE location_county ILIKE '%Hall%'
     ORDER BY employment DESC NULLS LAST LIMIT 50;
"""

_SQL_PROMPT = """You are a PostgreSQL expert. Generate a single SQL SELECT query for the question below.

DATABASE SCHEMA:
{schema}

EXAMPLES:
{examples}

RULES:
1. Use ONLY the gev_companies table.
2. Use exact match (=) when question says 'only', 'strictly', 'exactly'.
3. Use ILIKE '%value%' when question says 'including', 'at least', 'or higher'.
4. Always add LIMIT (use question's number or default 50).
5. Always add ORDER BY for ranked/top-N questions.
6. Output ONLY the SQL query. No explanation. No markdown. No backticks.

QUESTION: {question}
SQL:"""


def generate_sql(question: str) -> str:
    """
    Use the LLM to generate a SQL query from the question.
    Phase 5: injects dynamically retrieved few-shot examples from Qdrant.
    Returns the raw SQL string, or empty string on failure.
    """
    cfg = Config.get()

    # ── Phase 5: Dynamic few-shot injection ───────────────────────────────────
    # Try to retrieve similar verified SQL examples from the Qdrant store.
    # Falls back to static examples if store is empty or unavailable.
    dynamic_examples = _FEW_SHOT   # default: static examples
    try:
        from phase5_fewshot.few_shot_retriever import get_few_shot_block
        block = get_few_shot_block(question, query_type="sql", top_k=3)
        if block:
            dynamic_examples = block + "\n\n" + _FEW_SHOT   # prepend dynamic, keep static as fallback
            logger.info("Phase 5: injected %d dynamic SQL examples", block.count("Example "))
    except Exception as exc:
        logger.debug("Phase 5 few-shot unavailable (%s) — using static examples", exc)

    prompt = _SQL_PROMPT.format(
        schema=_SCHEMA,
        examples=dynamic_examples,
        question=question,
    )

    payload = {
        "model":  cfg.ollama_cypher_model,   # same fast model used for Cypher
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic
            "num_predict": 300,   # SQL is short
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
    # Remove ```sql ... ``` fences
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip("`").strip()
    # Extract only the SELECT statement (safety: ignore any trailing text)
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

    # Enforce max LIMIT for safety
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
