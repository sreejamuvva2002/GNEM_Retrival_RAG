"""
migrate_and_reload.py — Full data reload from GNEM Excel into PostgreSQL.

What this does:
  1. Adds facility_type column to gev_companies if missing
  2. Reloads ALL companies from GNEM Excel (applies employment overrides from CSV)
  3. Verifies all key columns have data

Employment overrides (global headcount → Georgia facility headcount) are now
in kb/employment_overrides.csv — edit that file, NOT this script.

Run:
  venv\\Scripts\\python scripts\\migrate_and_reload.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.db import get_engine, get_session, Company, create_tables
from shared.logger import get_logger
from phase1_extraction.kb_loader import load_companies_from_excel

logger = get_logger("migrate_and_reload")


def add_facility_type_column():
    """Add facility_type column if it doesn't exist yet."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='gev_companies' AND column_name='facility_type'"
        ))
        if result.fetchone():
            print("  facility_type column already exists ✅")
        else:
            conn.execute(text(
                "ALTER TABLE gev_companies ADD COLUMN facility_type VARCHAR(200)"
            ))
            conn.commit()
            print("  facility_type column added ✅")


def reload_all_companies():
    """
    Reload all companies from Excel into PostgreSQL.
    Employment overrides are applied automatically by kb_loader
    from kb/employment_overrides.csv — not hardcoded here.
    """
    print("\nLoading companies from Excel (with employment overrides from CSV)...")
    companies = load_companies_from_excel()
    print(f"  Loaded {len(companies)} companies from Excel")

    session = get_session()
    updated = 0
    created = 0

    try:
        for data in companies:
            name = data.get("company_name")
            if not name:
                continue

            company = session.query(Company).filter(
                Company.company_name == name
            ).first()

            if company:
                for field, value in data.items():
                    if hasattr(company, field) and value is not None:
                        setattr(company, field, value)
                updated += 1
            else:
                company = Company(**{k: v for k, v in data.items() if hasattr(Company, k)})
                session.add(company)
                created += 1

        session.commit()
        print(f"\n  Updated: {updated} companies")
        print(f"  Created: {created} new companies")

    except Exception as e:
        session.rollback()
        logger.error("Reload failed: %s", e)
        raise
    finally:
        session.close()


def verify_columns():
    """Verify all key columns have data after reload."""
    session = get_session()
    try:
        total = session.query(Company).count()
        print(f"\n{'='*55}")
        print(f"  Verification — {total} total companies")
        print(f"{'='*55}")

        for col_name, col in [
            ("ev_battery_relevant",      Company.ev_battery_relevant),
            ("industry_group",           Company.industry_group),
            ("supplier_affiliation_type",Company.supplier_affiliation_type),
            ("facility_type",            Company.facility_type),
            ("classification_method",    Company.classification_method),
        ]:
            filled = session.query(Company).filter(col.isnot(None)).count()
            sample = [r[0] for r in session.query(col).filter(col.isnot(None)).distinct().limit(3).all()]
            print(f"  {col_name:30s}: {filled}/{total} | sample: {sample}")

        # Verify employment overrides applied (spot-check)
        print(f"\n  Employment spot-check (should NOT show global headcount):")
        spot = ["Yamaha Motor Manufacturing Corp.", "Woory Industrial Co.", "Yazaki North America"]
        for name in spot:
            c = session.query(Company).filter(Company.company_name == name).first()
            if c:
                flag = "✅" if (c.employment or 0) < 10000 else "❌ STILL WRONG"
                print(f"    {name[:45]:45s}: {c.employment} {flag}")

    finally:
        session.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MIGRATE & RELOAD — Fresh Data from GNEM Excel")
    print("="*60)

    print("\n[1/3] Adding facility_type column...")
    add_facility_type_column()

    print("\n[2/3] Reloading all company data from Excel...")
    reload_all_companies()

    print("\n[3/3] Verifying data...")
    verify_columns()

    print("\n✅ Done. Next steps:")
    print("   venv\\Scripts\\python -m phase3_graph.pipeline")
    print("   venv\\Scripts\\python scripts\\sync_neo4j.py")
    print("   venv\\Scripts\\python scripts\\smoke_test_phase4.py")
test_phase4.py")
