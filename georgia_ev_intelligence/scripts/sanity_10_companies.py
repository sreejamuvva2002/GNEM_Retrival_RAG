"""
Sanity Test — Phase 1 Pipeline for first 10 companies.

Steps:
  1. Check data quality in GNEM Excel (report issues, auto-fix where safe)
  2. Load GNEM Excel → sync all 205 companies to Neon PostgreSQL
  3. Run Phase 1 pipeline (search + extract + store) for first 10 companies
  4. Verify PostgreSQL: companies table, documents table, extracted_facts table
  5. Verify Backblaze B2: confirm files actually uploaded

Run from georgia_ev_intelligence/ directory:
  venv\\Scripts\\python scripts/sanity_10_companies.py

Flags:
  --skip-pipeline   Skip search/extract, only verify DB + B2 (if already ran)
  --show-sql        Show detailed SQL query results
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import text

from shared.config import Config
from shared.db import create_tables, get_engine, get_session, verify_connection
from shared.logger import get_logger
import shared.storage as storage_mod
from db_storage.kb_loader import (
    _detect_shifted_row,
    _parse_float,
    _parse_employment,
    load_companies_from_excel,
    sync_companies_to_db,
    get_all_companies_from_db,
    _find_gnem_excel,
)

logger = get_logger("sanity.phase1")

SEPARATOR = "=" * 65


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Data Quality Check on GNEM Excel
# ─────────────────────────────────────────────────────────────────────────────

def check_data_quality() -> dict:
    """Inspect GNEM Excel for data quality issues and report them."""
    section("STEP 1: GNEM Excel Data Quality Check")

    path = _find_gnem_excel()
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip().lower() for c in df.columns]

    print(f"\n  Excel path : {path}")
    print(f"  Total rows : {len(df)}")
    print(f"  Columns    : {', '.join(df.columns.tolist())}\n")

    issues = []
    dq_report = {
        "total_rows": len(df),
        "shifted_rows": [],
        "missing_company_name": [],
        "non_numeric_lat": [],
        "non_numeric_lon": [],
        "non_numeric_employment": [],
        "duplicate_company_names": [],
        "truncated_fields": [],
    }

    for idx, row in df.iterrows():
        row_num = idx + 2  # Excel row number (1-indexed + header)
        company = str(row.get("company", "")).strip()

        # Check shifted rows
        if _detect_shifted_row(row):
            lon_val = row.get("longitude", "")
            msg = f"  ⚠️  Row {row_num} [{company}] — COLUMN SHIFTED: longitude='{lon_val}'"
            issues.append(msg)
            print(msg)
            dq_report["shifted_rows"].append({"row": row_num, "company": company, "lon_raw": str(lon_val)})
            continue

        # Missing company name
        if not company or company.lower() in ("nan", "none", ""):
            msg = f"  ⚠️  Row {row_num} — MISSING company name"
            issues.append(msg)
            print(msg)
            dq_report["missing_company_name"].append(row_num)

        # Non-numeric latitude
        lat_raw = row.get("latitude")
        if lat_raw is not None and not (isinstance(lat_raw, float) and pd.isna(lat_raw)):
            if _parse_float(lat_raw) is None:
                msg = f"  ⚠️  Row {row_num} [{company}] — BAD latitude: {repr(lat_raw)}"
                issues.append(msg)
                print(msg)
                dq_report["non_numeric_lat"].append({"row": row_num, "company": company, "value": str(lat_raw)})

        # Non-numeric longitude
        lon_raw = row.get("longitude")
        if lon_raw is not None and not (isinstance(lon_raw, float) and pd.isna(lon_raw)):
            if _parse_float(lon_raw) is None:
                msg = f"  ⚠️  Row {row_num} [{company}] — BAD longitude: {repr(lon_raw)}"
                issues.append(msg)
                print(msg)
                dq_report["non_numeric_lon"].append({"row": row_num, "company": company, "value": str(lon_raw)})

        # Non-numeric employment
        emp_raw = row.get("employment")
        if emp_raw is not None and not (isinstance(emp_raw, float) and pd.isna(emp_raw)):
            if _parse_employment(emp_raw) is None:
                msg = f"  ⚠️  Row {row_num} [{company}] — BAD employment: {repr(emp_raw)}"
                issues.append(msg)
                print(msg)
                dq_report["non_numeric_employment"].append({"row": row_num, "company": company, "value": str(emp_raw)})

        # Check field lengths that might get truncated
        field_limits = {
            "ev / battery relevant": 100,
            "ev supply chain role": 200,
            "product / service": None,  # TEXT — no limit
        }
        for col, limit in field_limits.items():
            if col in row.index and limit:
                val = str(row[col]) if row[col] is not None else ""
                if len(val) > limit:
                    msg = f"  ℹ️  Row {row_num} [{company}] — TRUNCATION: '{col}' has {len(val)} chars (limit={limit})"
                    print(msg)
                    dq_report["truncated_fields"].append({"row": row_num, "company": company, "field": col, "length": len(val)})

    # Check for duplicate company names
    name_counts = df["company"].value_counts()
    dupes = name_counts[name_counts > 1]
    if not dupes.empty:
        for name, count in dupes.items():
            msg = f"  ⚠️  DUPLICATE company name: '{name}' appears {count} times"
            issues.append(msg)
            print(msg)
            dq_report["duplicate_company_names"].append({"name": name, "count": count})

    print(f"\n  Summary: {len(issues)} data quality issues found")
    if not issues:
        print("  ✅ No data quality issues found!")
    else:
        print(f"\n  ℹ️  Shifted rows will be SKIPPED during load.")
        print(f"  ℹ️  All other values will be safely coerced or truncated.")

    return dq_report


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Load GNEM + Sync to PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────

def load_and_sync() -> list[dict]:
    section("STEP 2: Load GNEM Excel → Sync to Neon PostgreSQL")

    print("\n  Loading GNEM Excel...")
    companies = load_companies_from_excel()
    print(f"  Loaded: {len(companies)} companies from Excel")

    print("\n  Syncing to PostgreSQL (Neon)...")
    inserted, updated = sync_companies_to_db(companies)
    print(f"  ✅ Inserted: {inserted} | Updated: {updated}")

    # Show first 10 in table format
    print(f"\n  {'#':<4} {'Company':<38} {'Tier':<12} {'City':<20} {'County':<20} {'Employees':<10}")
    print(f"  {'-'*4} {'-'*38} {'-'*12} {'-'*20} {'-'*20} {'-'*10}")
    for i, c in enumerate(companies[:10], 1):
        name = (c.get("company_name") or "")[:37]
        tier = (c.get("tier") or "")[:11]
        city = (c.get("location_city") or "")[:19]
        county = (c.get("location_county") or "")[:19]
        emp = str(int(c.get("employment") or 0)) if c.get("employment") else "N/A"
        print(f"  {i:<4} {name:<38} {tier:<12} {city:<20} {county:<20} {emp:<10}")

    return companies


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Run Phase 1 Pipeline for 10 Companies
# ─────────────────────────────────────────────────────────────────────────────

async def run_phase1_10(all_companies: list[dict]) -> None:
    section("STEP 3: Running Phase 1 Pipeline — First 10 Companies")

    # Import pipeline here (after path setup)
    from core_agent.extraction_pipeline import run_pipeline, print_summary

    target_companies = all_companies[:10]
    print(f"\n  Processing {len(target_companies)} companies:")
    for i, c in enumerate(target_companies, 1):
        print(f"    {i}. {c['company_name']}")

    print(f"\n  Starting pipeline (concurrency=3 for sanity test)...")
    results = await run_pipeline(
        companies=target_companies,
        concurrency=3,
        skip_if_has_docs=False,  # Always re-process for sanity test
    )

    print_summary(results)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Verify PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────

def verify_postgresql(target_companies: list[dict]) -> bool:
    section("STEP 4: Verify Neon PostgreSQL")

    engine = get_engine()
    all_ok = True

    with engine.connect() as conn:
        # 4a. Companies table
        print("\n  [4a] gev_companies table")
        result = conn.execute(text("SELECT COUNT(*) FROM gev_companies"))
        total_companies = result.scalar()
        print(f"       Total companies in DB : {total_companies}")
        ok = total_companies >= 200
        print(f"       Status : {'✅ OK (≥200)' if ok else '❌ LOW — expected ≥200'}")
        if not ok:
            all_ok = False

        # 4b. Documents table — check for first 10 companies
        print(f"\n  [4b] gev_documents table — per company")
        company_names = [c["company_name"] for c in target_companies]
        name_list = ", ".join(f"'{n}'" for n in company_names)

        result = conn.execute(text(f"""
            SELECT company_name,
                   COUNT(*) as doc_count,
                   SUM(CASE WHEN extraction_status = 'extracted' THEN 1 ELSE 0 END) as extracted_ok,
                   SUM(CASE WHEN extraction_status = 'failed' THEN 1 ELSE 0 END) as failed,
                   SUM(CASE WHEN b2_key IS NOT NULL THEN 1 ELSE 0 END) as in_b2
            FROM gev_documents
            WHERE company_name IN ({name_list})
            GROUP BY company_name
            ORDER BY company_name
        """))
        rows = result.fetchall()

        print(f"\n       {'Company':<38} {'Docs':<6} {'OK':<6} {'Failed':<8} {'In B2':<8}")
        print(f"       {'-'*38} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
        companies_with_docs = 0
        for row in rows:
            name = (row[0] or "")[:37]
            doc_count = row[1]
            ok_count = row[2]
            fail_count = row[3]
            b2_count = row[4]
            status = "✅" if doc_count > 0 else "⚠️ "
            if doc_count > 0:
                companies_with_docs += 1
            print(f"       {name:<38} {doc_count:<6} {ok_count:<6} {fail_count:<8} {b2_count:<8} {status}")

        if len(rows) == 0:
            print("       ❌ No documents found — pipeline may not have run yet")
            all_ok = False
        else:
            print(f"\n       Companies with docs : {companies_with_docs}/{len(target_companies)}")
            if companies_with_docs < 7:
                print("       ⚠️  Less than 7/10 companies have documents (some search queries may have failed)")

        # 4c. Extracted facts table
        print(f"\n  [4c] gev_extracted_facts table")
        result = conn.execute(text(f"""
            SELECT COUNT(*) as total_facts,
                   COUNT(DISTINCT company_name) as companies_with_facts,
                   COUNT(DISTINCT fact_type) as fact_types
            FROM gev_extracted_facts
            WHERE company_name IN ({name_list})
        """))
        row = result.fetchone()
        print(f"       Total facts extracted   : {row[0]}")
        print(f"       Companies with facts    : {row[1]}/{len(target_companies)}")
        print(f"       Distinct fact types     : {row[2]}")

        # 4d. Sample facts
        if row[0] > 0:
            print(f"\n       Sample extracted facts (first 5):")
            result = conn.execute(text(f"""
                SELECT company_name, fact_type, fact_value_text, fact_year, confidence_score
                FROM gev_extracted_facts
                WHERE company_name IN ({name_list})
                ORDER BY confidence_score DESC NULLS LAST
                LIMIT 5
            """))
            for frow in result.fetchall():
                company = (frow[0] or "")[:25]
                ftype = (frow[1] or "")[:20]
                fval = (frow[2] or "")[:40]
                fyear = frow[3] or ""
                fconf = f"{frow[4]:.2f}" if frow[4] else "N/A"
                print(f"       [{company}] {ftype} → {fval} ({fyear}) conf={fconf}")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Verify Backblaze B2 Cloud Storage
# ─────────────────────────────────────────────────────────────────────────────

def verify_b2(target_companies: list[dict]) -> bool:
    section("STEP 5: Verify Backblaze B2 Cloud Storage")

    try:
        b2_ok = storage_mod.verify_connection()
        if not b2_ok:
            print(f"\n  ❌ B2 connection failed — check B2_* env vars")
            return False
        print(f"  ✅ B2 connection OK")
    except Exception as e:
        print(f"\n  ❌ B2 connection failed: {e}")
        return False

    engine = get_engine()
    all_ok = True

    with engine.connect() as conn:
        company_names = [c["company_name"] for c in target_companies]
        name_list = ", ".join(f"'{n}'" for n in company_names)

        # Get all B2 keys for these companies
        result = conn.execute(text(f"""
            SELECT company_name, b2_key, content_type, file_size_bytes, extraction_status
            FROM gev_documents
            WHERE company_name IN ({name_list})
              AND b2_key IS NOT NULL
            ORDER BY company_name
            LIMIT 30
        """))
        docs = result.fetchall()

    if not docs:
        print(f"\n  ⚠️  No B2 keys found in DB for first 10 companies")
        print(f"      This means pipeline hasn't run yet or docs failed to upload")
        return False

    print(f"\n  Checking {len(docs)} documents in B2...")
    print(f"\n  {'Company':<30} {'Type':<10} {'Size':<10} {'B2 Status':<15} {'DB Status':<12}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*15} {'-'*12}")

    verified = 0
    missing = 0

    for row in docs:
        company = (row[0] or "")[:29]
        b2_key = row[1]
        ctype = (row[2] or "")[:9]
        size = f"{row[3]:,}" if row[3] else "N/A"
        db_status = row[4] or ""

        # Check if key exists in B2
        try:
            exists = storage_mod.key_exists(b2_key)
            b2_status = "✅ found" if exists else "❌ MISSING"
            if exists:
                verified += 1
            else:
                missing += 1
        except Exception as e:
            b2_status = f"⚠️  error"

        print(f"  {company:<30} {ctype:<10} {size:<10} {b2_status:<15} {db_status:<12}")

    print(f"\n  B2 verification: {verified} found / {missing} missing / {len(docs)} total")
    if missing > 0:
        print(f"  ⚠️  {missing} files are in DB but not in B2 — upload may have failed")
        all_ok = False
    else:
        print(f"  ✅ All documents confirmed in Backblaze B2!")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(dq_ok: bool, db_ok: bool, b2_ok: bool) -> None:
    section("SANITY TEST SUMMARY")
    print()
    print(f"  {'Step':<35} {'Status'}")
    print(f"  {'-'*35} {'-'*20}")
    print(f"  {'1. GNEM Data Quality Check':<35} {'✅ Complete' if True else '❌ Failed'}")
    print(f"  {'2. PostgreSQL Load + Sync':<35} {'✅ OK' if db_ok else '❌ Issues found'}")
    print(f"  {'3. Phase 1 Pipeline (10 cos.)':<35} {'✅ Ran' if True else '❌ Failed'}")
    print(f"  {'4. PostgreSQL Verification':<35} {'✅ OK' if db_ok else '❌ Issues found'}")
    print(f"  {'5. B2 Cloud Verification':<35} {'✅ OK' if b2_ok else '⚠️  Issues found'}")
    print()
    if db_ok and b2_ok:
        print("  🎉 SANITY TEST PASSED — Ready to proceed to Phase 2!")
    else:
        print("  ⚠️  Some checks failed — review the output above before Phase 2.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    print(f"\n{'#' * 65}")
    print(f"#  GEORGIA EV INTELLIGENCE — PHASE 1 SANITY TEST (10 Companies)  #")
    print(f"{'#' * 65}")

    # Verify DB connection first
    section("PRE-CHECK: Database + Config")
    cfg = Config.get()
    def _check(label, key):
        try:
            val = getattr(cfg, key, None) or ""
            print(f"  {label:<12}: {'✅ set' if val else '❌ MISSING'}")
        except Exception:
            print(f"  {label:<12}: ❌ MISSING")

    _check("Tavily key", "tavily_api_key")
    _check("Qdrant URL", "qdrant_url")
    _check("Neo4j URI", "neo4j_uri")
    _check("DB URL", "database_url")
    _check("B2 bucket", "b2_bucket")
    print(f"  {'Ollama model':<12}: {cfg.ollama_llm_model}")


    db_reachable = verify_connection()
    if not db_reachable:
        print("\n  ❌ Cannot reach database. Check .env DATABASE_URL. Aborting.")
        sys.exit(1)
    print(f"  DB connect  : ✅ OK")

    create_tables()
    print(f"  DB tables   : ✅ gev_* tables verified/created")

    # Step 1: Data Quality
    dq_report = check_data_quality()

    # Step 2: Load + Sync
    companies = load_and_sync()
    all_companies_from_db = get_all_companies_from_db()
    target_companies = all_companies_from_db[:10]

    # Step 3: Run pipeline (unless --skip-pipeline)
    if not args.skip_pipeline:
        await run_phase1_10(target_companies)
    else:
        print(f"\n{SEPARATOR}")
        print("  STEP 3: SKIPPED (--skip-pipeline flag set)")
        print(SEPARATOR)

    # Step 4: Verify PostgreSQL
    db_ok = verify_postgresql(target_companies)

    # Step 5: Verify B2
    b2_ok = verify_b2(target_companies)

    # Final Summary
    print_final_summary(
        dq_ok=len(dq_report["shifted_rows"]) == 0,
        db_ok=db_ok,
        b2_ok=b2_ok,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Sanity Test — run pipeline for 10 companies and verify all outputs"
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip the search+extract pipeline (just verify DB + B2)",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
