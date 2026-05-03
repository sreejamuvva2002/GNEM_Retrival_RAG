"""
fresh_load.py — Complete fresh load: wipe everything, reload from Excel.

WHAT THIS DOES (in order):
  1. Wipe PostgreSQL table (gev_companies)
  2. Wipe Neo4j graph (all nodes + relationships)
  3. Load 193 clean companies from GNEM Excel → PostgreSQL
     (employment overrides applied from kb/employment_overrides.csv)
  4. Rebuild Neo4j graph from PostgreSQL
  5. Sync any remaining properties (facility_type) to Neo4j
  6. Print verification summary

WHY FRESH LOAD INSTEAD OF APPEND:
  - 205 rows in Excel, 193 loaded (12 skipped for data quality)
  - We know exactly which 12 are bad (column-shifted rows)
  - Append risks duplicates + broken Neo4j relationships
  - Fresh load gives 100% known state

Run:
  venv\\Scripts\\python scripts\\fresh_load.py

IMPORTANT: This DELETES all data first. Run only when you want a full reset.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.db import get_engine, get_session, Company, create_tables
from shared.logger import get_logger
from phase1_extraction.kb_loader import load_companies_from_excel
from phase3_graph.graph_loader import (
    get_driver, verify_connection, create_schema,
    load_companies, load_locations, load_oem_relationships,
    load_tier_relationships, load_industry_relationships,
    load_product_relationships, get_graph_stats, close_driver,
)
from phase1_extraction.kb_loader import get_all_companies_from_db
from sqlalchemy import text

logger = get_logger("scripts.fresh_load")
SEP = "=" * 60


# ── Step 1: Wipe PostgreSQL ────────────────────────────────────────────────────

def wipe_postgres() -> None:
    """
    Delete all rows from all gev_* tables in correct FK dependency order.

    FK chain (must delete children before parents):
      gev_document_chunks  →  gev_documents  →  gev_companies
      gev_extracted_facts  →  gev_documents
      gev_extracted_facts  →  gev_companies
      gev_eval_results     (no FK, but wipe for clean state)

    Using TRUNCATE ... CASCADE handles all of this atomically.
    """
    engine = get_engine()
    with engine.connect() as conn:
        # Count before
        count_before = conn.execute(text("SELECT COUNT(*) FROM gev_companies")).scalar()

        # TRUNCATE CASCADE: PostgreSQL automatically truncates all dependent tables
        # in one atomic operation — no FK violation possible.
        conn.execute(text(
            "TRUNCATE TABLE gev_companies, gev_documents, gev_document_chunks, "
            "gev_extracted_facts, gev_eval_results RESTART IDENTITY CASCADE"
        ))
        conn.commit()
    print(f"  Wiped gev_companies ({count_before} rows) + all dependent tables ✅")



# ── Step 2: Wipe Neo4j ────────────────────────────────────────────────────────

def wipe_neo4j() -> None:
    """Delete ALL nodes and relationships from Neo4j."""
    driver = get_driver()
    with driver.session() as s:
        # Count first
        node_count = s.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        rel_count  = s.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
        # Wipe in batches (avoids memory error on large graphs)
        s.run("CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'DETACH DELETE n', {batchSize:1000})")
    print(f"  Deleted {node_count} nodes, {rel_count} relationships from Neo4j ✅")


def wipe_neo4j_simple() -> None:
    """Simple wipe (no APOC) — works for small graphs like ours (~750 nodes)."""
    driver = get_driver()
    with driver.session() as s:
        node_count = s.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        rel_count  = s.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
        s.run("MATCH (n) DETACH DELETE n")
    print(f"  Deleted {node_count} nodes, {rel_count} relationships from Neo4j ✅")


def _deduplicate_companies(companies: list[dict]) -> list[dict]:
    """
    Deduplicate companies by company_name.

    WHY: Some companies appear multiple times in the Excel with different
    ev_supply_chain_role or products_services (e.g. ZF Gainesville LLC appears
    twice with 'EV body, powertrain...' and 'EV thermal management...').

    Strategy:
      - First occurrence wins for all scalar fields (tier, county, employment, etc.)
      - ev_supply_chain_role: concatenate unique values with ' | '
      - products_services: concatenate unique values with ' | '
    """
    seen: dict[str, dict] = {}
    dupes: list[str] = []

    for company in companies:
        name = (company.get("company_name") or "").strip()
        if not name:
            continue

        if name not in seen:
            seen[name] = company.copy()
        else:
            dupes.append(name)
            existing = seen[name]

            # Merge ev_supply_chain_role
            old_role = str(existing.get("ev_supply_chain_role") or "").strip()
            new_role = str(company.get("ev_supply_chain_role") or "").strip()
            if new_role and new_role not in old_role:
                existing["ev_supply_chain_role"] = f"{old_role} | {new_role}".strip(" |")

            # Merge products_services
            old_prod = str(existing.get("products_services") or "").strip()
            new_prod = str(company.get("products_services") or "").strip()
            if new_prod and new_prod not in old_prod:
                existing["products_services"] = f"{old_prod} | {new_prod}".strip(" |")

    if dupes:
        print(f"  ℹ️  Merged {len(dupes)} duplicate entries: {dupes}")

    return list(seen.values())


def load_postgres() -> int:
    """
    Load companies from Excel into PostgreSQL.
    Deduplicates by company_name before inserting.
    Inserts row-by-row so one bad row never kills the whole batch.
    Returns count of successfully loaded companies.
    """
    create_tables()
    print("\n  Loading from GNEM Excel (employment overrides applied from CSV)...")
    raw_companies = load_companies_from_excel()
    print(f"  Raw rows from Excel : {len(raw_companies)}")

    companies = _deduplicate_companies(raw_companies)
    print(f"  After deduplication : {len(companies)} unique companies")

    loaded = 0
    skipped = 0

    for data in companies:
        name = data.get("company_name")
        if not name:
            skipped += 1
            continue

        # Safety: cast float columns
        for fc in ("latitude", "longitude", "employment"):
            v = data.get(fc)
            if v is not None:
                try:
                    data[fc] = float(v)
                except (ValueError, TypeError):
                    data[fc] = None

        # Per-row session: one failure doesn't kill others
        session = get_session()
        try:
            company = Company(**{k: v for k, v in data.items() if hasattr(Company, k)})
            session.add(company)
            session.commit()
            loaded += 1
        except Exception as exc:
            session.rollback()
            logger.warning("Skipped '%s': %s", name, str(exc)[:80])
            skipped += 1
        finally:
            session.close()

    print(f"  Loaded  : {loaded} companies ✅")
    if skipped:
        print(f"  Skipped : {skipped} companies (see warnings above)")
    return loaded



# ── Step 4: Build Neo4j graph ─────────────────────────────────────────────────

def build_neo4j() -> dict:
    """Build Neo4j graph from PostgreSQL. Returns stats."""
    create_schema()
    companies = get_all_companies_from_db()
    print(f"\n  Building graph from {len(companies)} companies...")

    t0 = time.monotonic()
    load_companies(companies)
    load_locations(companies)
    load_oem_relationships(companies)
    load_tier_relationships(companies)
    load_industry_relationships(companies)
    load_product_relationships(companies)
    elapsed = time.monotonic() - t0

    stats = get_graph_stats()
    print(f"  Graph built in {elapsed:.1f}s")
    print(f"  Nodes: {stats.get('total_nodes', 0)} | Relationships: {stats.get('total_rels', 0)} ✅")
    return stats


# ── Step 5: Sync properties to Neo4j ─────────────────────────────────────────

def sync_properties_to_neo4j() -> None:
    """Push facility_type, supplier_affiliation_type, employment caps to Neo4j."""
    session = get_session()
    try:
        companies = session.query(Company).all()
    finally:
        session.close()

    driver = get_driver()
    synced = 0
    with driver.session() as s:
        for c in companies:
            name = (c.company_name or "").strip()
            if not name:
                continue
            s.run(
                """
                MATCH (node:Company {name: $name})
                SET node.facility_type             = $facility_type,
                    node.supplier_affiliation_type = $affiliation_type,
                    node.employment                = $employment,
                    node.synced_at                 = datetime()
                """,
                name=name,
                facility_type=str(getattr(c, "facility_type", "") or ""),
                affiliation_type=str(c.supplier_affiliation_type or ""),
                employment=float(c.employment) if c.employment else None,
            )
            synced += 1
    print(f"  Synced {synced} company properties to Neo4j ✅")


# ── Verification ──────────────────────────────────────────────────────────────

def verify_all() -> None:
    """Print a summary verification of both databases."""
    print(f"\n{SEP}")
    print("  VERIFICATION")
    print(SEP)

    # PostgreSQL
    session = get_session()
    try:
        pg_count = session.query(Company).count()
        overridden = session.query(Company).filter(Company.employment < 10000).count()
        print(f"\n  PostgreSQL:")
        print(f"    Total companies : {pg_count}")

        # Spot check employment
        checks = [
            "Yamaha Motor Manufacturing Corp.",
            "Woory Industrial Co.",
            "Yazaki North America",
        ]
        for name in checks:
            c = session.query(Company).filter(Company.company_name == name).first()
            if c:
                flag = "✅" if (c.employment or 0) < 10000 else "❌ STILL GLOBAL HEADCOUNT"
                print(f"    {name[:45]:45s}: {c.employment} {flag}")
    finally:
        session.close()

    # Neo4j
    driver = get_driver()
    with driver.session() as s:
        node_count = s.run("MATCH (n:Company) RETURN count(n) AS cnt").single()["cnt"]
        with_fac   = s.run(
            "MATCH (c:Company) WHERE c.facility_type IS NOT NULL AND c.facility_type <> '' "
            "RETURN count(c) AS cnt"
        ).single()["cnt"]
    print(f"\n  Neo4j:")
    print(f"    Company nodes           : {node_count}")
    print(f"    Nodes with facility_type: {with_fac}")

    # Final verdict
    print(f"\n  {'='*40}")
    if pg_count >= 190 and node_count >= 190 and with_fac >= 180:
        print("  ✅ ALL SYSTEMS READY — run smoke test next")
        print("     venv\\Scripts\\python scripts\\smoke_test_phase4.py")
    else:
        print("  ⚠️  Something looks off — check counts above")
    print(f"  {'='*40}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{SEP}")
    print("  FRESH LOAD — Complete Database Reset + Reload")
    print(f"{SEP}")
    print("  ⚠️  This will DELETE all PostgreSQL + Neo4j data first.\n")

    # Verify Neo4j is reachable before wiping anything
    check = verify_connection()
    if not check["ok"]:
        print(f"  ❌ Neo4j connection FAILED: {check['error']}")
        print("     Fix Neo4j connection before running fresh load.")
        sys.exit(1)
    print("  Neo4j connection: ✅\n")

    # STEP 1: Wipe
    print(f"[1/5] Wiping PostgreSQL...")
    wipe_postgres()

    print(f"\n[2/5] Wiping Neo4j...")
    try:
        wipe_neo4j_simple()
    except Exception as e:
        print(f"  Warning: Neo4j wipe issue: {e}")
        print("  Continuing — existing nodes will be overwritten by MERGE")

    # STEP 3: Load PostgreSQL
    print(f"\n[3/5] Loading PostgreSQL from GNEM Excel...")
    pg_count = load_postgres()

    # STEP 4: Build Neo4j
    print(f"\n[4/5] Building Neo4j graph...")
    stats = build_neo4j()

    # STEP 5: Sync extra properties
    print(f"\n[5/5] Syncing extra properties to Neo4j...")
    sync_properties_to_neo4j()

    # Verify
    verify_all()

    close_driver()
