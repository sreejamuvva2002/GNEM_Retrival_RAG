"""
Phase 3 — Neo4j Knowledge Graph

WHY A GRAPH alongside Qdrant?
  Qdrant answers: "Find chunks semantically similar to this query"
  Neo4j answers:  "Which Tier 1 suppliers in Hall County supply Hyundai?"

  These are fundamentally different query types:
    - Vector search = similarity (fuzzy, semantic)
    - Graph traversal = relationships (exact, structural)

  Example where graph wins:
    Q: "List all Tier 2 suppliers that supply to a Tier 1 company
        that supplies to Hyundai METAPLANT in Bryan County"
    Qdrant: cannot answer this (no relationship traversal)
    Neo4j:  MATCH (t2:Company {tier:'Tier 2'})-[:SUPPLIES_TO]->
                  (t1:Company {tier:'Tier 1'})-[:SUPPLIES_TO]->
                  (oem:OEM {name:'Hyundai'})
            WHERE t1.location_county = 'Bryan County'
            RETURN t2.name

SCHEMA (6 lean node types — fits within AuraDB Free 200k/400k limit):
  Nodes:
    (:Company)     — 193 nodes (one per GNEM row)
    (:Location)    — ~120 unique city+county combos
    (:OEM)         — ~25 OEM names (Hyundai, Kia, Ford, etc.)
    (:Tier)        — 5 tiers (OEM, Tier 1, Tier 2, Tier 3, Indirect)
    (:IndustryGroup) — ~30 unique industry groups
    (:Product)     — ~400 product/service entries

  Relationships:
    (Company)-[:LOCATED_IN]->(Location)
    (Company)-[:SUPPLIES_TO]->(OEM)
    (Company)-[:IN_TIER]->(Tier)
    (Company)-[:IN_INDUSTRY]->(IndustryGroup)
    (Company)-[:MANUFACTURES]->(Product)
    (Company)-[:PEER_OF]->(Company)  [same tier + same OEM]

TOTAL ESTIMATE: ~750 nodes, ~3,500 relationships
AuraDB Free limit: 200,000 nodes / 400,000 relationships
We use < 0.4% of limit.
"""
from __future__ import annotations

import re
from typing import Any

from neo4j import GraphDatabase, Driver
from shared.config import Config
from shared.logger import get_logger

logger = get_logger("phase3.graph_loader")

_driver: Driver | None = None


def get_driver() -> Driver:
    """Singleton Neo4j driver."""
    global _driver
    if _driver is None:
        cfg = Config.get()
        _driver = GraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_username, cfg.neo4j_password),
        )
        logger.info("Neo4j driver created: %s", cfg.neo4j_uri)
    return _driver


def close_driver() -> None:
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def verify_connection() -> dict[str, Any]:
    """Verify Neo4j AuraDB is reachable."""
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run("RETURN 'connected' AS status, 1+1 AS math")
            row = result.single()
            return {"ok": True, "status": row["status"], "math": row["math"]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA — Constraints + Indexes
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_STATEMENTS = [
    # Uniqueness constraints (also create indexes automatically)
    "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.location_id IS UNIQUE",
    "CREATE CONSTRAINT oem_name IF NOT EXISTS FOR (o:OEM) REQUIRE o.name IS UNIQUE",
    "CREATE CONSTRAINT tier_name IF NOT EXISTS FOR (t:Tier) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT industry_name IF NOT EXISTS FOR (i:IndustryGroup) REQUIRE i.name IS UNIQUE",
    # Additional indexes for common filter fields
    "CREATE INDEX company_tier IF NOT EXISTS FOR (c:Company) ON (c.tier)",
    "CREATE INDEX company_county IF NOT EXISTS FOR (c:Company) ON (c.location_county)",
    "CREATE INDEX company_ev_relevant IF NOT EXISTS FOR (c:Company) ON (c.ev_battery_relevant)",
    "CREATE INDEX company_employment IF NOT EXISTS FOR (c:Company) ON (c.employment)",
]


def create_schema() -> None:
    """Create constraints and indexes. Safe to run multiple times (IF NOT EXISTS)."""
    driver = get_driver()
    with driver.session() as session:
        for stmt in SCHEMA_STATEMENTS:
            try:
                session.run(stmt)
            except Exception as exc:
                logger.warning("Schema statement skipped: %s — %s", stmt[:60], exc)
    logger.info("Neo4j schema created/verified (%d statements)", len(SCHEMA_STATEMENTS))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_oems(primary_oems: str | None) -> list[str]:
    """
    Parse OEM string like "Hyundai, Kia, BMW" → ["Hyundai", "Kia", "BMW"]
    Handles: commas, slashes, semicolons, 'and', 'N/A', None
    """
    if not primary_oems or str(primary_oems).strip().lower() in ("n/a", "none", "nan", ""):
        return []
    raw = str(primary_oems)
    parts = re.split(r"[,;/&\n]|\band\b", raw, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 2]


def _parse_products(products_services: str | None) -> list[str]:
    """
    Parse products/services into a list.
    Truncates long entries, skips empty ones.
    Returns at most 5 products per company (keeps graph lean).
    """
    if not products_services or str(products_services).strip().lower() in ("n/a", "none", "nan", ""):
        return []
    raw = str(products_services).strip()
    # Split on comma or semicolon
    parts = re.split(r"[,;]", raw)
    products = []
    for p in parts:
        p = p.strip()[:100]  # Cap at 100 chars
        if p and len(p) >= 3:
            products.append(p)
    return products[:5]  # Max 5 per company


def _safe_str(val: Any, max_len: int = 500) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in ("nan", "none", "n/a", ""):
        return None
    return s[:max_len]


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(str(val).replace(",", ""))
        return f if f > 0 else None
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# NODE + RELATIONSHIP LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_companies(companies: list[dict[str, Any]]) -> int:
    """
    MERGE all Company nodes.
    Uses MERGE (not CREATE) so re-runs are idempotent.
    """
    driver = get_driver()
    loaded = 0

    with driver.session() as session:
        for company in companies:
            name = _safe_str(company.get("company_name"))
            if not name:
                continue
            try:
                session.run(
                    """
                    MERGE (c:Company {name: $name})
                    SET c.tier                      = $tier,
                        c.ev_supply_chain_role      = $role,
                        c.ev_battery_relevant       = $ev_relevant,
                        c.industry_group            = $industry,
                        c.facility_type             = $facility_type,
                        c.supplier_affiliation_type = $affiliation_type,
                        c.location_city             = $city,
                        c.location_county           = $county,
                        c.location_state            = 'Georgia',
                        c.employment                = $employment,
                        c.products_services         = $products,
                        c.primary_oems              = $oems_raw,
                        c.latitude                  = $lat,
                        c.longitude                 = $lng,
                        c.classification            = $classification,
                        c.gnem_id                   = $gnem_id,
                        c.updated_at                = datetime()
                    """,
                    name=name,
                    tier=_safe_str(company.get("tier")),
                    role=_safe_str(company.get("ev_supply_chain_role")),
                    ev_relevant=_safe_str(company.get("ev_battery_relevant")),
                    industry=_safe_str(company.get("industry_group")),
                    facility_type=_safe_str(company.get("facility_type")),
                    affiliation_type=_safe_str(company.get("supplier_affiliation_type")),
                    city=_safe_str(company.get("location_city")),
                    county=_safe_str(company.get("location_county")),
                    employment=_safe_float(company.get("employment")),
                    products=_safe_str(company.get("products_services")),
                    oems_raw=_safe_str(company.get("primary_oems")),
                    lat=_safe_float(company.get("latitude")),
                    lng=_safe_float(company.get("longitude")),
                    classification=_safe_str(company.get("classification_method")),
                    gnem_id=company.get("id"),
                )
                loaded += 1
            except Exception as exc:
                logger.warning("Failed to load company '%s': %s", name, exc)

    logger.info("Loaded %d/%d company nodes", loaded, len(companies))
    return loaded


def load_locations(companies: list[dict[str, Any]]) -> int:
    """Create Location nodes and LOCATED_IN relationships."""
    driver = get_driver()
    created = 0

    with driver.session() as session:
        for company in companies:
            name   = _safe_str(company.get("company_name"))
            city   = _safe_str(company.get("location_city"))
            county = _safe_str(company.get("location_county"))
            if not name or not county:
                continue

            location_id = f"{(city or '').lower().replace(' ', '_')}_{county.lower().replace(' ', '_')}"

            try:
                session.run(
                    """
                    MERGE (l:Location {location_id: $lid})
                    SET l.city   = $city,
                        l.county = $county,
                        l.state  = 'Georgia'
                    WITH l
                    MATCH (c:Company {name: $company_name})
                    MERGE (c)-[:LOCATED_IN]->(l)
                    """,
                    lid=location_id,
                    city=city,
                    county=county,
                    company_name=name,
                )
                created += 1
            except Exception as exc:
                logger.warning("Location error for '%s': %s", name, exc)

    logger.info("Created location nodes + LOCATED_IN relationships for %d companies", created)
    return created


def load_oem_relationships(companies: list[dict[str, Any]]) -> int:
    """Create OEM nodes and SUPPLIES_TO relationships."""
    driver = get_driver()
    created = 0

    with driver.session() as session:
        for company in companies:
            name = _safe_str(company.get("company_name"))
            if not name:
                continue
            oems = _parse_oems(company.get("primary_oems"))
            for oem_name in oems:
                try:
                    session.run(
                        """
                        MERGE (o:OEM {name: $oem_name})
                        WITH o
                        MATCH (c:Company {name: $company_name})
                        MERGE (c)-[:SUPPLIES_TO]->(o)
                        """,
                        oem_name=oem_name,
                        company_name=name,
                    )
                    created += 1
                except Exception as exc:
                    logger.warning("OEM error '%s' → '%s': %s", name, oem_name, exc)

    logger.info("Created %d OEM nodes + SUPPLIES_TO relationships", created)
    return created


def load_tier_relationships(companies: list[dict[str, Any]]) -> int:
    """Create Tier nodes and IN_TIER relationships."""
    driver = get_driver()
    created = 0

    with driver.session() as session:
        for company in companies:
            name = _safe_str(company.get("company_name"))
            tier = _safe_str(company.get("tier"))
            if not name or not tier:
                continue
            try:
                session.run(
                    """
                    MERGE (t:Tier {name: $tier})
                    WITH t
                    MATCH (c:Company {name: $company_name})
                    MERGE (c)-[:IN_TIER]->(t)
                    """,
                    tier=tier,
                    company_name=name,
                )
                created += 1
            except Exception as exc:
                logger.warning("Tier error '%s': %s", name, exc)

    logger.info("Created tier nodes + IN_TIER relationships for %d companies", created)
    return created


def load_industry_relationships(companies: list[dict[str, Any]]) -> int:
    """Create IndustryGroup nodes and IN_INDUSTRY relationships."""
    driver = get_driver()
    created = 0

    with driver.session() as session:
        for company in companies:
            name     = _safe_str(company.get("company_name"))
            industry = _safe_str(company.get("industry_group"))
            if not name or not industry:
                continue
            try:
                session.run(
                    """
                    MERGE (i:IndustryGroup {name: $industry})
                    WITH i
                    MATCH (c:Company {name: $company_name})
                    MERGE (c)-[:IN_INDUSTRY]->(i)
                    """,
                    industry=industry,
                    company_name=name,
                )
                created += 1
            except Exception as exc:
                logger.warning("Industry error '%s': %s", name, exc)

    logger.info("Created industry nodes + IN_INDUSTRY relationships for %d companies", created)
    return created


def load_product_relationships(companies: list[dict[str, Any]]) -> int:
    """Create Product nodes and MANUFACTURES relationships. Max 5 products per company."""
    driver = get_driver()
    created = 0

    with driver.session() as session:
        for company in companies:
            name     = _safe_str(company.get("company_name"))
            products = _parse_products(company.get("products_services"))
            if not name or not products:
                continue
            for product in products:
                # Product node ID = normalized product name
                product_id = re.sub(r"\s+", "_", product.lower())[:80]
                try:
                    session.run(
                        """
                        MERGE (p:Product {product_id: $pid})
                        SET p.name = $product_name
                        WITH p
                        MATCH (c:Company {name: $company_name})
                        MERGE (c)-[:MANUFACTURES]->(p)
                        """,
                        pid=product_id,
                        product_name=product,
                        company_name=name,
                    )
                    created += 1
                except Exception as exc:
                    logger.warning("Product error '%s' → '%s': %s", name, product, exc)

    logger.info("Created %d product nodes + MANUFACTURES relationships", created)
    return created


def get_graph_stats() -> dict[str, Any]:
    """Return counts of all nodes and relationships in the graph."""
    driver = get_driver()
    stats = {}
    try:
        with driver.session() as session:
            node_types = ["Company", "Location", "OEM", "Tier", "IndustryGroup", "Product"]
            for label in node_types:
                count = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt").single()["cnt"]
                stats[f"nodes_{label.lower()}"] = count

            rel_types = ["LOCATED_IN", "SUPPLIES_TO", "IN_TIER", "IN_INDUSTRY", "MANUFACTURES"]
            for rel in rel_types:
                count = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS cnt").single()["cnt"]
                stats[f"rels_{rel.lower()}"] = count

            total_nodes = sum(v for k, v in stats.items() if k.startswith("nodes_"))
            total_rels  = sum(v for k, v in stats.items() if k.startswith("rels_"))
            stats["total_nodes"] = total_nodes
            stats["total_rels"]  = total_rels
    except Exception as exc:
        stats["error"] = str(exc)
    return stats
