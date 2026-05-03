"""
phase4_agent/text_to_cypher.py
==============================================================
Text-to-Cypher: converts any natural language question into
a Cypher query using the local Gemma model, then executes it
against Neo4j with one self-healing retry on failure.

WHY GEMMA FOR CYPHER GENERATION:
  - Faster than qwen2.5:14b (~1-3s vs 5-10s)
  - Google trained it with strong code understanding
  - Smaller context needed (just schema + question)
  - qwen2.5:14b is reserved for full answer synthesis

WHY THIS IS GENERIC (vs old keyword approach):
  - No stop words, no bigrams, no hardcoded entity extraction
  - LLM reads the schema and generates the correct Cypher
  - Works for any new question automatically
  - Multi-hop traversal is native Cypher (impossible in SQL)

SCHEMA NOTE:
  Neo4j Company nodes use c.name (NOT c.company_name)
  All queries must alias: c.name AS company_name
"""
from __future__ import annotations

import re
import httpx
from phase3_graph.graph_loader import get_driver
from shared.config import Config
from shared.logger import get_logger

logger = get_logger("phase4.text_to_cypher")

# ── Neo4j Schema description (given to LLM for context) ──────────────────────
_SCHEMA = """
GRAPH SCHEMA — Georgia EV Supply Chain

NODE LABELS AND PROPERTIES:
  (:Company)
    name                    STRING  — company name (PRIMARY KEY, use for MATCH)
    tier                    STRING  — "Tier 1", "Tier 1/2", "Tier 2/3", "OEM",
                                      "OEM Supply Chain", "OEM Footprint"
    ev_supply_chain_role    STRING  — "Battery Cell", "Battery Pack",
                                      "Thermal Management", "Power Electronics",
                                      "Charging Infrastructure", "Vehicle Assembly",
                                      "General Automotive", "Materials"
    ev_battery_relevant     STRING  — "Yes", "No", "Indirect"
    industry_group          STRING  — e.g. "Primary Metal Industries",
                                      "Chemicals and Allied Products"
    facility_type           STRING  — "Manufacturing Plant", "R&D", "Headquarters",
                                      "Distribution Center", "Assembly"
    supplier_affiliation_type STRING — "Automotive supply chain participant", etc.
    location_county         STRING  — county name, e.g. "Gwinnett County"
    location_city           STRING  — city name
    employment              FLOAT   — number of employees at Georgia facility
    products_services       STRING  — free-text product description
    primary_oems            STRING  — comma-separated OEM names

  (:OEM)    name STRING
  (:Tier)   name STRING
  (:Location)   city STRING, county STRING, state STRING
  (:IndustryGroup)  name STRING
  (:Product)    name STRING

RELATIONSHIPS:
  (Company)-[:SUPPLIES_TO]->(:OEM)
  (Company)-[:LOCATED_IN]->(:Location)
  (Company)-[:IN_TIER]->(:Tier)
  (Company)-[:IN_INDUSTRY]->(:IndustryGroup)
  (Company)-[:MANUFACTURES]->(:Product)
  (Company)-[:PEER_OF]->(Company)   [same tier + same OEM set]

IMPORTANT:
  - Company node property is 'name' (not 'company_name')
  - Always alias: c.name AS company_name in RETURN
  - Use toLower() + CONTAINS for case-insensitive partial matching
  - Use IS NOT NULL checks before CONTAINS to avoid null pointer errors
"""

# ── Few-shot examples (question → Cypher) ────────────────────────────────────
_EXAMPLES = """
-- Q: Which companies supply Rivian?
MATCH (c:Company)-[:SUPPLIES_TO]->(o:OEM)
WHERE toLower(o.name) CONTAINS 'rivian'
RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
       c.employment, c.location_county
ORDER BY c.tier, c.employment DESC
LIMIT 50

-- Q: Find all Tier 1/2 companies
MATCH (c:Company)-[:IN_TIER]->(t:Tier)
WHERE t.name CONTAINS 'Tier 1'
RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
       c.location_county, c.employment
ORDER BY c.employment DESC
LIMIT 50

-- Q: Which companies manufacture copper foil or battery materials?
MATCH (c:Company)
WHERE (c.products_services IS NOT NULL AND toLower(c.products_services) CONTAINS 'copper')
   OR (c.products_services IS NOT NULL AND toLower(c.products_services) CONTAINS 'battery material')
RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
       c.location_county, c.employment, c.products_services
ORDER BY c.employment DESC
LIMIT 50

-- Q: Companies in Gwinnett County with highest employment
MATCH (c:Company)-[:LOCATED_IN]->(l:Location)
WHERE toLower(l.county) CONTAINS 'gwinnett'
RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
       c.employment, l.county AS location_county
ORDER BY c.employment DESC
LIMIT 20

-- Q: Which roles have only one supplier (single point of failure)?
MATCH (c:Company)
WHERE c.ev_supply_chain_role IS NOT NULL
WITH c.ev_supply_chain_role AS role, collect(c.name) AS companies
WHERE size(companies) = 1
RETURN role, companies[0] AS company_name
ORDER BY role

-- Q: Show Tier 1 suppliers connected to both Rivian and Hyundai
MATCH (c:Company)-[:SUPPLIES_TO]->(o1:OEM)
MATCH (c)-[:SUPPLIES_TO]->(o2:OEM)
WHERE toLower(o1.name) CONTAINS 'rivian'
  AND toLower(o2.name) CONTAINS 'hyundai'
RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
       c.location_county, c.employment

-- Q: Find R&D facilities in Georgia
MATCH (c:Company)
WHERE c.facility_type IS NOT NULL AND toLower(c.facility_type) CONTAINS 'r&d'
RETURN c.name AS company_name, c.tier, c.location_county,
       c.employment, c.facility_type
ORDER BY c.employment DESC

-- Q: Which Battery Cell or Battery Pack companies are Tier 1/2?
MATCH (c:Company)-[:IN_TIER]->(t:Tier)
WHERE (t.name CONTAINS 'Tier 1')
  AND (c.ev_supply_chain_role IS NOT NULL)
  AND (c.ev_supply_chain_role CONTAINS 'Battery Cell'
       OR c.ev_supply_chain_role CONTAINS 'Battery Pack')
RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
       c.location_county, c.employment
ORDER BY c.employment DESC

-- Q: Vehicle Assembly OEMs and their Tier 1 suppliers
MATCH (assembler:Company)-[:SUPPLIES_TO]->(oem:OEM)
MATCH (supplier:Company)-[:SUPPLIES_TO]->(oem)
WHERE assembler.ev_supply_chain_role IS NOT NULL
  AND assembler.ev_supply_chain_role CONTAINS 'Vehicle Assembly'
  AND supplier.tier IS NOT NULL
  AND supplier.tier CONTAINS 'Tier 1'
RETURN assembler.name AS oem_company,
       oem.name AS oem_name,
       collect(DISTINCT supplier.name) AS tier1_suppliers
ORDER BY assembler.name

-- Q: Companies involved in battery recycling
MATCH (c:Company)
WHERE (c.name IS NOT NULL AND toLower(c.name) CONTAINS 'recycl')
   OR (c.products_services IS NOT NULL AND toLower(c.products_services) CONTAINS 'recycl')
RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
       c.location_county, c.employment, c.products_services
ORDER BY c.employment DESC
"""

# ── Cypher generation prompt ──────────────────────────────────────────────────
_CYPHER_PROMPT = """You are a Neo4j Cypher expert for the Georgia EV Supply Chain Intelligence System.

GRAPH SCHEMA:
{schema}

EXAMPLE QUESTION-TO-CYPHER PAIRS:
{examples}

STRICT RULES:
1. Return ONLY the raw Cypher query — NO explanation, NO markdown, NO backticks
2. ALWAYS alias c.name AS company_name in the RETURN clause
3. Use toLower() + CONTAINS for string matching (case-insensitive)
4. Always add IS NOT NULL check before CONTAINS on nullable fields
5. Use LIMIT 50 to prevent huge result sets
6. For multi-hop: MATCH multiple patterns on the same variable c
7. For product/text search: search c.products_services (full text on node)
   AND also (c)-[:MANUFACTURES]->(p:Product) when appropriate
8. For county: match both c.location_county directly AND via [:LOCATED_IN]->(l:Location)
9. NEVER use c.company_name — the property is c.name

QUESTION: {question}

CYPHER:"""


def generate_cypher(question: str, error_feedback: str = "") -> str:
    """
    Use Gemma (fast, code-focused) to generate a Cypher query.
    If error_feedback is provided, includes it for self-correction.
    """
    cfg = Config.get()
    prompt_question = question
    if error_feedback:
        prompt_question = (
            f"{question}\n\n"
            f"PREVIOUS ATTEMPT FAILED WITH ERROR:\n{error_feedback}\n"
            f"Fix the Cypher query to avoid this error."
        )

    prompt = _CYPHER_PROMPT.format(
        schema=_SCHEMA,
        examples=_EXAMPLES,
        question=prompt_question,
    )

    payload = {
        "model": cfg.ollama_cypher_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,     # deterministic — we want exact Cypher
            "num_predict": 400,     # Cypher queries are short
            "num_ctx": 4096,
        },
    }

    try:
        url = f"{cfg.ollama_base_url}/api/generate"
        with httpx.Client(timeout=90.0) as client:   # 90s — generous for cold start
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            raw = str(resp.json().get("response", "")).strip()
            # Strip any accidental markdown fences
            cypher = _clean_cypher(raw)
            logger.info("Generated Cypher (%d chars): %s", len(cypher), cypher[:100])
            return cypher
    except Exception as exc:
        logger.error("Cypher generation failed: %s", exc)
        return ""


def _clean_cypher(raw: str) -> str:
    """Remove markdown fences, leading/trailing whitespace, thinking tags."""
    # Remove <think>...</think> blocks (some models emit these)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Remove ```cypher ... ``` or ``` ... ```
    raw = re.sub(r"```(?:cypher)?", "", raw, flags=re.IGNORECASE).strip("`").strip()
    # Remove any explanatory text before the first MATCH/WITH/RETURN
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
        # If it's a connection drop, reset the driver so next call reconnects
        if any(kw in err_str.lower() for kw in ("routing", "connection", "no data", "defunct")):
            logger.info("Neo4j connection lost — resetting driver for reconnect")
            from phase3_graph.graph_loader import close_driver
            try:
                close_driver()
            except Exception:
                pass
        raise  # re-raise so caller can do self-heal


def execute_cypher_safe(question: str) -> list[dict]:
    """
    Generate Cypher and execute with one self-healing retry.
    If generation fails → return []
    If execution fails → regenerate with error message → retry once
    If retry also fails → return []

    IMPORTANT: Python 3 deletes the `exc` variable after an except block exits.
    We must capture the error message INSIDE the except block.
    """
    # Attempt 1
    cypher = generate_cypher(question)
    if not cypher:
        logger.warning("Cypher generation produced empty string")
        return []

    error_msg = ""   # ← capture here so it's accessible after the except block
    try:
        return execute_cypher(cypher)
    except Exception as exc:
        error_msg = str(exc)   # ← save BEFORE except block exits (Python 3 deletes exc)
        logger.warning("Cypher attempt 1 failed, self-healing: %s", error_msg)

    # Attempt 2 — self-heal: send the error back to Gemma to fix the query
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
    Normalize Neo4j result rows to the standard _company_to_dict format
    so they work with the existing _format_companies() pipeline method.
    """
    normalized = []
    for row in rows:
        # company_name: always aliased as company_name in our queries
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
