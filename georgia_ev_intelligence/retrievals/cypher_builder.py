"""
phase4_agent/cypher_builder.py
==============================================================
Deterministic Cypher builder — NO LLM, NO hardcoded domain words.

DESIGN PRINCIPLE — "Fair game":
  Every entity used here was extracted from REAL database values
  by entity_extractor.py. Nothing is guessed from question keywords.

  entity_extractor loads at startup:
    - tier          → real Tier node names from Neo4j
    - county        → real location_county values from PostgreSQL
    - oem           → real OEM node names from Neo4j
    - ev_role_list  → real ev_supply_chain_role values from PostgreSQL
    - company_name  → real company_name values from PostgreSQL
    - facility_type → real facility_type values from PostgreSQL  ← NEW

  This file only builds Cypher query STRUCTURE using those entities.
  The structure follows the fixed Neo4j schema — that is acceptable
  because the schema itself is the source of truth.

WHAT IS NOT HARDCODED HERE:
  - No word-to-DB-value mappings (no facility_map, no role_map)
  - No question pattern matching
  - No domain-specific string literals

RETURNS None when no entity matched → pipeline falls through to Gemma.
"""
from __future__ import annotations

from filters_and_validation.query_entity_extractor import Entities
from shared.logger import get_logger

logger = get_logger("retrievals.cypher_builder")


def build_cypher(e: Entities, question: str) -> str | None:
    """
    Build a Cypher query from already-extracted entities.
    Every branch depends only on what entity_extractor found in the DB.
    Returns None if no entity was extracted (Gemma handles these).
    """

    # ── 1. OEM supplier network ───────────────────────────────────────────────
    # Triggered when entity_extractor found a real OEM name from Neo4j.
    if e.oem:
        cypher = f"""MATCH (c:Company)-[:SUPPLIES_TO]->(o:OEM)
WHERE toLower(o.name) CONTAINS '{e.oem.lower()}'
RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
       c.employment, c.location_county, c.facility_type,
       c.ev_battery_relevant, c.industry_group, c.products_services,
       o.name AS oem_name
ORDER BY c.tier, c.employment DESC
LIMIT 50"""
        logger.info("Cypher built (OEM path): oem=%s", e.oem)
        return cypher

    # ── 2. EV role filter ─────────────────────────────────────────────────────
    # Triggered when entity_extractor found real role values from PostgreSQL.
    if e.ev_role_list:
        role_conditions = " OR ".join(
            f"toLower(c.ev_supply_chain_role) CONTAINS '{r.lower()}'"
            for r in e.ev_role_list
        )
        tier_filter = ""
        if e.tier:
            tier_filter = f"\n  AND c.tier IS NOT NULL AND toLower(c.tier) CONTAINS '{e.tier.lower()}'"

        cypher = f"""MATCH (c:Company)
WHERE c.ev_supply_chain_role IS NOT NULL
  AND ({role_conditions}){tier_filter}
RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
       c.employment, c.location_county, c.facility_type,
       c.ev_battery_relevant, c.industry_group, c.products_services
ORDER BY c.employment DESC
LIMIT 50"""
        logger.info("Cypher built (Role path): roles=%s", e.ev_role_list)
        return cypher

    # ── 3. Facility type filter ───────────────────────────────────────────────
    # Triggered when entity_extractor matched a real facility_type value from
    # PostgreSQL — the candidate set is loaded from the live KB, not from a
    # hardcoded map.
    if e.facility_type:
        cypher = f"""MATCH (c:Company)
WHERE c.facility_type IS NOT NULL
  AND toLower(c.facility_type) CONTAINS '{e.facility_type.lower()}'
RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
       c.employment, c.location_county, c.facility_type,
       c.ev_battery_relevant, c.industry_group, c.products_services
ORDER BY c.employment DESC
LIMIT 50"""
        logger.info("Cypher built (Facility path): facility_type=%s", e.facility_type)
        return cypher

    # ── 4. Company direct lookup ──────────────────────────────────────────────
    # Triggered when entity_extractor matched a real company name from PostgreSQL.
    if e.company_name:
        cypher = f"""MATCH (c:Company)
WHERE toLower(c.name) CONTAINS '{e.company_name.lower()}'
RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
       c.employment, c.location_county, c.facility_type,
       c.ev_battery_relevant, c.industry_group, c.products_services,
       c.primary_oems
LIMIT 10"""
        logger.info("Cypher built (Company path): name=%s", e.company_name)
        return cypher

    # ── 5. Product keyword search — AND-first, OR-fallback ────────────────────
    # WHY AND-FIRST:
    #   OR across many keywords matches too many rows (false positives).
    #   Two-keyword AND is precise and survives noisy question wordings.
    #
    # STRATEGY (no stop words, no hardcoding, works for any new question):
    #   1. Take the first 2 keywords (most specific, per question word order)
    #   2. Build an AND Cypher — both must appear in products_services
    #   3. If AND returns 0 rows → fall back to OR across top 3 keywords
    #   This is stateless and scales to any domain without maintenance.
    if e.product_keywords:
        # Top 2 specific terms (question word order = most specific first)
        and_kw = e.product_keywords[:2]
        # Top 3 for OR fallback
        or_kw  = e.product_keywords[:3]

        and_conditions = " AND ".join(
            f"(c.products_services IS NOT NULL AND toLower(c.products_services) CONTAINS '{kw.lower()}')"
            for kw in and_kw
        )
        or_conditions = " OR ".join(
            f"(c.products_services IS NOT NULL AND toLower(c.products_services) CONTAINS '{kw.lower()}')"
            for kw in or_kw
        )
        name_conditions = " OR ".join(
            f"toLower(c.name) CONTAINS '{kw.lower()}'"
            for kw in or_kw
        )

        # AND Cypher — tried first by execute_cypher_safe in pipeline
        cypher = f"""MATCH (c:Company)
WHERE ({and_conditions})
RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,
       c.employment, c.location_county, c.facility_type,
       c.ev_battery_relevant, c.industry_group, c.products_services
ORDER BY c.employment DESC
LIMIT 50
// FALLBACK_OR: {or_conditions} OR ({name_conditions})"""
        logger.info("Cypher built (Product AND-first): and_kw=%s or_kw=%s", and_kw, or_kw)
        return cypher

    # ── Nothing matched → Gemma handles this question ────────────────────────
    logger.info("Cypher builder: no entity extracted, Gemma will handle this")
    return None
