"""
Phase 4 — Cypher Retriever
Runs Neo4j graph traversal queries for relationship-based questions.

WHY GRAPH FOR SOME QUESTIONS:
  SQL is flat. Graph traversal answers:
    "<tier> suppliers connected to <oem>" → follow SUPPLIES_TO edges
    "Which roles have only 1 supplier?"   → count relationship patterns
    "Companies in <county> + <tier>"      → multi-node pattern match
"""
from __future__ import annotations
from typing import Any
from db_storage.graph_loader import get_driver
from shared.logger import get_logger

logger = get_logger("retrievals.cypher_retriever")


def query_companies_by_tier(tier: str) -> list[dict]:
    """MATCH (c:Company)-[:IN_TIER]->(t:Tier {name: $tier})"""
    driver = get_driver()
    with driver.session() as s:
        rows = s.run(
            "MATCH (c:Company)-[:IN_TIER]->(t:Tier) "
            "WHERE toLower(t.name) CONTAINS toLower($tier) "
            "RETURN c.name AS company_name, c.tier AS tier, "
            "       c.ev_supply_chain_role AS ev_supply_chain_role, "
            "       c.employment AS employment, "
            "       c.location_county AS location_county, "
            "       c.products_services AS products_services "
            "ORDER BY c.name",
            tier=tier,
        ).data()
    logger.info("Cypher tier query '%s' -> %d results", tier, len(rows))
    return rows


def query_companies_by_location(county: str) -> list[dict]:
    """MATCH (c:Company)-[:LOCATED_IN]->(l:Location)"""
    driver = get_driver()
    with driver.session() as s:
        rows = s.run(
            "MATCH (c:Company)-[:LOCATED_IN]->(l:Location) "
            "WHERE toLower(l.county) CONTAINS toLower($county) "
            "RETURN c.name AS company_name, c.tier AS tier, "
            "       c.ev_supply_chain_role AS ev_supply_chain_role, "
            "       c.employment AS employment, "
            "       l.city AS location_city, l.county AS location_county "
            "ORDER BY c.employment DESC",
            county=county,
        ).data()
    logger.info("Cypher location query '%s' -> %d results", county, len(rows))
    return rows


def query_oem_suppliers(oem_name: str) -> list[dict]:
    """MATCH (c:Company)-[:SUPPLIES_TO]->(o:OEM)"""
    driver = get_driver()
    with driver.session() as s:
        rows = s.run(
            "MATCH (c:Company)-[:SUPPLIES_TO]->(o:OEM) "
            "WHERE toLower(o.name) CONTAINS toLower($oem_name) "
            "RETURN c.name AS company_name, c.tier AS tier, "
            "       c.ev_supply_chain_role AS ev_supply_chain_role, "
            "       c.employment AS employment, "
            "       c.location_county AS location_county, "
            "       o.name AS primary_oems "
            "ORDER BY c.tier, c.name",
            oem_name=oem_name,
        ).data()
    logger.info("Cypher OEM query '%s' -> %d results", oem_name, len(rows))
    return rows


    return rows


def query_single_supplier_roles() -> list[dict]:
    """Find EV roles served by exactly one company (single-point-of-failure)."""
    driver = get_driver()
    with driver.session() as s:
        rows = s.run(
            "MATCH (c:Company) "
            "WITH c.ev_supply_chain_role AS role, collect(c.name) AS companies "
            "WHERE size(companies) = 1 AND role IS NOT NULL "
            "RETURN role, companies[0] AS company "
            "ORDER BY role"
        ).data()
    logger.info("Single-supplier roles: %d found", len(rows))
    return rows


def query_county_tier_density() -> list[dict]:
    """Return company count per county+tier combination."""
    driver = get_driver()
    with driver.session() as s:
        rows = s.run(
            "MATCH (c:Company)-[:LOCATED_IN]->(l:Location) "
            "RETURN l.county AS county, c.tier AS tier, count(c) AS company_count "
            "ORDER BY company_count DESC"
        ).data()
    return rows


def query_graph_stats() -> dict[str, Any]:
    """Return overall graph statistics."""
    driver = get_driver()
    with driver.session() as s:
        result = s.run(
            "MATCH (c:Company) RETURN count(c) AS companies "
            "UNION ALL MATCH ()-[r]->() RETURN count(r) AS companies"
        ).data()
    return {"node_count": result[0]["companies"], "rel_count": result[1]["companies"]}


# ── Pipeline-level wiring: CypherPlan + run_plan ─────────────────────────────

def _wrap_rows_as_candidates(rows: list[dict]) -> list:
    """Convert Cypher rows into Candidate objects for the fusion layer."""
    from core_agent.retrieval_types import Candidate
    out: list = []
    for r in rows:
        name = (r.get("company_name") or r.get("company") or "").strip()
        if not name:
            continue
        cand = Candidate(canonical_name=name, row=r)
        cand.add_source("cypher", 1.0)
        out.append(cand)
    return out


def run_plan(plan) -> list:
    """
    Execute a CypherPlan and wrap rows as Candidates.

    Modes:
      - tier            : query_companies_by_tier(plan.tier)
      - location        : query_companies_by_location(plan.county)
      - oem_network     : query_oem_suppliers(plan.oem_name)
      - single_supplier : query_single_supplier_roles()
      - county_density  : query_county_tier_density() (aggregate; raw rows)
    """
    mode = plan.mode

    if mode == "tier" and plan.tier:
        return _wrap_rows_as_candidates(query_companies_by_tier(plan.tier))

    if mode == "location" and plan.county:
        return _wrap_rows_as_candidates(query_companies_by_location(plan.county))

    if mode == "oem_network" and plan.oem_name:
        return _wrap_rows_as_candidates(query_oem_suppliers(plan.oem_name))

    if mode == "single_supplier":
        from core_agent.retrieval_types import Candidate
        rows = query_single_supplier_roles()
        out: list = []
        for r in rows:
            name = (r.get("company") or r.get("company_name") or r.get("role") or "").strip()
            if not name:
                continue
            cand = Candidate(canonical_name=name, row=r)
            cand.add_source("cypher", 1.0)
            out.append(cand)
        return out

    if mode == "county_density":
        from core_agent.retrieval_types import Candidate
        rows = query_county_tier_density()
        out: list = []
        for r in rows:
            cand = Candidate(
                canonical_name=f"{r.get('county')} / {r.get('tier')} (count)",
                row=r,
            )
            cand.add_source("cypher", 1.0)
            out.append(cand)
        return out

    logger.warning("Cypher run_plan: unhandled mode=%s tier=%s county=%s oem=%s",
                   mode, plan.tier, plan.county, plan.oem_name)
    return []


def render_cypher_query(plan) -> str:
    """
    Return the human-readable Cypher template string used for a plan.
    Stored in the audit log so reviewers can reproduce the query.
    The actual execution uses parameterised toLower() CONTAINS clauses.
    """
    if plan.mode == "tier":
        return ("MATCH (c:Company)-[:IN_TIER]->(t:Tier) "
                "WHERE toLower(t.name) CONTAINS toLower($tier) RETURN c, t")
    if plan.mode == "location":
        return ("MATCH (c:Company)-[:LOCATED_IN]->(l:Location) "
                "WHERE toLower(l.county) CONTAINS toLower($county) RETURN c, l")
    if plan.mode == "oem_network":
        return ("MATCH (c:Company)-[:SUPPLIES_TO]->(o:OEM) "
                "WHERE toLower(o.name) CONTAINS toLower($oem_name) RETURN c, o")
    if plan.mode == "single_supplier":
        return ("MATCH (c:Company) WITH c.ev_supply_chain_role AS role, "
                "collect(c.name) AS companies WHERE size(companies) = 1 "
                "AND role IS NOT NULL RETURN role, companies[0] AS company")
    if plan.mode == "county_density":
        return ("MATCH (c:Company)-[:LOCATED_IN]->(l:Location) "
                "RETURN l.county, c.tier, count(c)")
    return ""
