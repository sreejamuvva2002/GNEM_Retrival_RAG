"""
sync_neo4j.py — Sync Neo4j Company nodes from PostgreSQL (source of truth)

WHY THIS IS NEEDED:
  - PostgreSQL has capped employment (Yamaha=1500 not 190k, Woory=500 not 87k)
  - PostgreSQL has facility_type column (added after initial Neo4j load)
  - PostgreSQL has supplier_affiliation_type
  - Neo4j was loaded ONCE from the raw Excel — it has stale/wrong values

This script reads all companies from PostgreSQL and updates every Neo4j
Company node's properties to match exactly. Run after any PostgreSQL data change.

Usage:
    venv\\Scripts\\python scripts\\sync_neo4j.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# ── Add project root to path ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.db import get_session, Company
from db_storage.graph_loader import get_driver
from shared.logger import get_logger

logger = get_logger("scripts.sync_neo4j")


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(str(val).replace(",", ""))
        return f if f > 0 else None
    except (ValueError, TypeError):
        return None


def sync_companies_to_neo4j() -> int:
    """Read all companies from PostgreSQL and update Neo4j Company nodes."""
    session = get_session()
    try:
        companies = session.query(Company).all()
        logger.info("Loaded %d companies from PostgreSQL", len(companies))
    finally:
        session.close()

    driver = get_driver()
    updated = 0
    skipped = 0

    with driver.session() as neo4j_session:
        for c in companies:
            name = (c.company_name or "").strip()
            if not name:
                skipped += 1
                continue
            try:
                # MERGE on name (unique constraint), then SET all properties
                neo4j_session.run(
                    """
                    MERGE (node:Company {name: $name})
                    SET node.tier                      = $tier,
                        node.ev_supply_chain_role      = $role,
                        node.ev_battery_relevant       = $ev_relevant,
                        node.industry_group            = $industry,
                        node.facility_type             = $facility_type,
                        node.supplier_affiliation_type = $affiliation_type,
                        node.location_city             = $city,
                        node.location_county           = $county,
                        node.location_state            = 'Georgia',
                        node.employment                = $employment,
                        node.products_services         = $products,
                        node.primary_oems              = $oems,
                        node.classification            = $classification,
                        node.synced_at                 = datetime()
                    """,
                    name=name,
                    tier=str(c.tier or ""),
                    role=str(c.ev_supply_chain_role or ""),
                    ev_relevant=str(c.ev_battery_relevant or ""),
                    industry=str(c.industry_group or ""),
                    facility_type=str(getattr(c, "facility_type", "") or ""),
                    affiliation_type=str(c.supplier_affiliation_type or ""),
                    city=str(c.location_city or ""),
                    county=str(c.location_county or ""),
                    employment=_safe_float(c.employment),
                    products=str(c.products_services or "")[:500],
                    oems=str(c.primary_oems or ""),
                    classification=str(c.classification_method or ""),
                )
                updated += 1
            except Exception as exc:
                logger.warning("Failed to sync '%s': %s", name, exc)
                skipped += 1

    logger.info("Neo4j sync complete: %d updated, %d skipped", updated, skipped)
    return updated


def verify_sync() -> None:
    """Quick verification — compare counts and spot-check a few nodes."""
    driver = get_driver()
    with driver.session() as s:
        total = s.run("MATCH (c:Company) RETURN count(c) AS cnt").single()["cnt"]
        with_facility = s.run(
            "MATCH (c:Company) WHERE c.facility_type IS NOT NULL AND c.facility_type <> '' "
            "RETURN count(c) AS cnt"
        ).single()["cnt"]
        sample = s.run(
            "MATCH (c:Company) WHERE c.employment IS NOT NULL "
            "RETURN c.name, c.employment, c.facility_type "
            "ORDER BY c.employment DESC LIMIT 5"
        ).data()

    print(f"\n{'='*60}")
    print(f"  Neo4j Sync Verification")
    print(f"{'='*60}")
    print(f"  Total Company nodes:        {total}")
    print(f"  Nodes with facility_type:   {with_facility}")
    print(f"\n  Top 5 by employment (should NOT show 190k or 87k):")
    for row in sample:
        print(f"    {row['c.name'][:40]:40s} | emp={row['c.employment']} | facility={row['c.facility_type']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NEO4J SYNC FROM POSTGRESQL")
    print("="*60)
    print("\n[1/2] Syncing company properties...")
    count = sync_companies_to_neo4j()
    print(f"  ✅ Synced {count} companies")
    print("\n[2/2] Verifying sync...")
    verify_sync()
    print("  Done! Neo4j is now in sync with PostgreSQL.\n")
