"""
Phase 3 — Graph Build Pipeline

Loads all 193 companies from PostgreSQL into Neo4j AuraDB.

Run:
  venv\\Scripts\\python -m core_agent.graph_pipeline

Expected output after run:
  ~750 nodes, ~3,500 relationships in Neo4j
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

from db_storage.graph_loader import (
    verify_connection,
    create_schema,
    load_companies,
    load_locations,
    load_oem_relationships,
    load_tier_relationships,
    load_industry_relationships,
    load_product_relationships,
    get_graph_stats,
    close_driver,
)
from db_storage.kb_loader import get_all_companies_from_db
from shared.logger import get_logger

logger = get_logger("core_agent.pipeline")
SEP = "=" * 55


def main() -> None:
    print(f"\n{SEP}")
    print("GEORGIA EV — PHASE 3: NEO4J GRAPH BUILD")
    print(SEP)

    # 1. Verify connection
    logger.info("Verifying Neo4j connection...")
    check = verify_connection()
    if not check["ok"]:
        logger.error("Neo4j connection FAILED: %s", check["error"])
        sys.exit(1)
    logger.info("✅ Neo4j connected")

    # 2. Create schema (constraints + indexes)
    logger.info("Creating schema...")
    create_schema()

    # 3. Load companies from PostgreSQL
    logger.info("Loading companies from DB...")
    companies = get_all_companies_from_db()
    logger.info("Found %d companies to load", len(companies))

    start = time.monotonic()

    # 4. Load nodes and relationships in order
    # Company nodes first (all others reference them)
    n_companies = load_companies(companies)

    # Then relationships (MERGE finds existing Company nodes)
    n_locations = load_locations(companies)
    n_oems      = load_oem_relationships(companies)
    n_tiers     = load_tier_relationships(companies)
    n_industries = load_industry_relationships(companies)
    n_products  = load_product_relationships(companies)

    elapsed = time.monotonic() - start

    # 5. Final stats
    stats = get_graph_stats()

    print(f"\n{SEP}")
    print("PHASE 3 COMPLETE")
    print(SEP)
    print(f"  Time elapsed     : {elapsed:.1f}s")
    print(f"\n  NODES:")
    print(f"    Company        : {stats.get('nodes_company', 0)}")
    print(f"    Location       : {stats.get('nodes_location', 0)}")
    print(f"    OEM            : {stats.get('nodes_oem', 0)}")
    print(f"    Tier           : {stats.get('nodes_tier', 0)}")
    print(f"    IndustryGroup  : {stats.get('nodes_industrygroup', 0)}")
    print(f"    Product        : {stats.get('nodes_product', 0)}")
    print(f"    TOTAL          : {stats.get('total_nodes', 0)}")
    print(f"\n  RELATIONSHIPS:")
    print(f"    LOCATED_IN     : {stats.get('rels_located_in', 0)}")
    print(f"    SUPPLIES_TO    : {stats.get('rels_supplies_to', 0)}")
    print(f"    IN_TIER        : {stats.get('rels_in_tier', 0)}")
    print(f"    IN_INDUSTRY    : {stats.get('rels_in_industry', 0)}")
    print(f"    MANUFACTURES   : {stats.get('rels_manufactures', 0)}")
    print(f"    TOTAL          : {stats.get('total_rels', 0)}")
    print(SEP)

    close_driver()


if __name__ == "__main__":
    main()
