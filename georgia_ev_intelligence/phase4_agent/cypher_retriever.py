"""
Phase 4 — Cypher Retriever
Runs Neo4j graph traversal queries for relationship-based questions.

WHY GRAPH FOR SOME QUESTIONS:
  SQL is flat. Graph traversal answers:
    "Show Tier 1 suppliers connected to Rivian" → follow SUPPLIES_TO edges
    "Which roles have only 1 supplier?" → count relationship patterns
    "Companies in Hall County + Tier 1" → multi-node pattern match
"""
from __future__ import annotations
from typing import Any
from phase3_graph.graph_loader import get_driver
from shared.logger import get_logger

logger = get_logger("phase4.cypher_retriever")


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
