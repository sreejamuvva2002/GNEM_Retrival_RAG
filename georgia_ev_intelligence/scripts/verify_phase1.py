"""
Phase 1 Verification Script
Checks PostgreSQL (Neon) + Backblaze B2 after running the pipeline.

Usage:
  venv\Scripts\python scripts\verify_phase1.py

Shows:
  - Companies in DB
  - Documents extracted per company
  - Structured facts per company
  - B2 cloud files uploaded
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from shared.db import get_engine, get_session
from shared.db import Company, Document, ExtractedFact
from shared.storage import get_b2_client
from shared.config import Config

SEP = "=" * 70


def check_companies(session) -> int:
    """Show all companies with document counts."""
    print(f"\n{SEP}")
    print("COMPANIES IN NEON DB (gev_companies)")
    print(SEP)

    rows = session.execute(text("""
        SELECT
            c.id,
            c.company_name,
            c.tier,
            c.location_city,
            c.location_county,
            c.employment,
            COUNT(d.id) AS doc_count,
            COUNT(f.id) AS fact_count
        FROM gev_companies c
        LEFT JOIN gev_documents d ON d.company_id = c.id AND d.extraction_status = 'extracted'
        LEFT JOIN gev_extracted_facts f ON f.company_id = c.id
        GROUP BY c.id, c.company_name, c.tier, c.location_city, c.location_county, c.employment
        ORDER BY doc_count DESC, c.company_name
        LIMIT 20
    """)).fetchall()

    print(f"{'ID':>4}  {'Company':<35} {'Tier':<12} {'Location':<20} {'Docs':>4} {'Facts':>5}")
    print("-" * 84)
    for row in rows:
        location = f"{row.location_city or ''}, {row.location_county or ''}".strip(", ")
        print(
            f"{row.id:>4}  {row.company_name[:35]:<35} "
            f"{(row.tier or '')[:12]:<12} {location[:20]:<20} "
            f"{row.doc_count:>4} {row.fact_count:>5}"
        )

    total = session.execute(text("SELECT COUNT(*) FROM gev_companies")).scalar()
    print(f"\nTotal in DB: {total} companies")
    return total


def check_documents(session) -> None:
    """Show extracted documents grouped by company."""
    print(f"\n{SEP}")
    print("DOCUMENTS IN NEON DB (gev_documents)")
    print(SEP)

    # Overall stats
    stats = session.execute(text("""
        SELECT
            extraction_status,
            content_type,
            COUNT(*) AS cnt,
            SUM(word_count) AS total_words,
            SUM(file_size_bytes) AS total_bytes
        FROM gev_documents
        GROUP BY extraction_status, content_type
        ORDER BY extraction_status, content_type
    """)).fetchall()

    print(f"{'Status':<12} {'Type':<8} {'Count':>6} {'Total Words':>12} {'Total KB':>10}")
    print("-" * 52)
    for r in stats:
        kb = (r.total_bytes or 0) / 1024
        print(
            f"{(r.extraction_status or 'unknown')[:12]:<12} {(r.content_type or '')[:8]:<8} "
            f"{r.cnt:>6} {(r.total_words or 0):>12,} {kb:>10.1f}"
        )

    # Recent documents
    recent = session.execute(text("""
        SELECT company_name, content_type, word_count, document_type, source_url, b2_key
        FROM gev_documents
        WHERE extraction_status = 'extracted'
        ORDER BY id DESC
        LIMIT 15
    """)).fetchall()

    print(f"\n{'Company':<30} {'Type':<6} {'Words':>6} {'DocType':<20} {'B2 Key'}")
    print("-" * 100)
    for r in recent:
        b2_short = (r.b2_key or "no-b2-key")[-50:] if r.b2_key else "—"
        print(
            f"{(r.company_name or '')[:30]:<30} {(r.content_type or '')[:6]:<6} "
            f"{(r.word_count or 0):>6} {(r.document_type or '')[:20]:<20} {b2_short}"
        )


def check_facts(session) -> None:
    """Show structured facts extracted."""
    print(f"\n{SEP}")
    print("STRUCTURED FACTS IN NEON DB (gev_extracted_facts)")
    print(SEP)

    # By fact type
    by_type = session.execute(text("""
        SELECT fact_type, COUNT(*) AS cnt, AVG(confidence_score) AS avg_conf
        FROM gev_extracted_facts
        GROUP BY fact_type
        ORDER BY cnt DESC
    """)).fetchall()

    print(f"{'Fact Type':<25} {'Count':>6} {'Avg Confidence':>15}")
    print("-" * 50)
    for r in by_type:
        print(f"{r.fact_type[:25]:<25} {r.cnt:>6} {r.avg_conf:>15.2f}")

    # Top investment facts
    top_facts = session.execute(text("""
        SELECT company_name, fact_type, fact_value_text, fact_year, confidence_score
        FROM gev_extracted_facts
        WHERE fact_type IN ('investment', 'jobs_created', 'jobs_total')
        ORDER BY fact_value_numeric DESC NULLS LAST
        LIMIT 10
    """)).fetchall()

    if top_facts:
        print(f"\nTop Investment / Jobs Facts:")
        print(f"{'Company':<30} {'Type':<15} {'Value':<20} {'Year':>5} {'Conf':>5}")
        print("-" * 80)
        for r in top_facts:
            print(
                f"{(r.company_name or '')[:30]:<30} {r.fact_type[:15]:<15} "
                f"{(r.fact_value_text or '')[:20]:<20} {r.fact_year or '':>5} {r.confidence_score or 0:>5.2f}"
            )


def check_b2() -> None:
    """List files in Backblaze B2 bucket."""
    print(f"\n{SEP}")
    print("BACKBLAZE B2 CLOUD STORAGE")
    print(SEP)

    try:
        cfg = Config.get()
        client = get_b2_client()
        response = client.list_objects_v2(
            Bucket=cfg.b2_bucket,           # correct: cfg.b2_bucket not cfg.b2_bucket_name
            Prefix="documents/",
            MaxKeys=50,
        )
        objects = response.get("Contents", [])

        if not objects:
            print("  No files found in B2 under documents/")
            print("  (Run the pipeline first to upload documents)")
            return

        print(f"Found {len(objects)} files (showing up to 50):\n")
        print(f"{'B2 Key':<65} {'Size KB':>8} {'Modified'}")
        print("-" * 90)
        total_kb = 0
        for obj in objects:
            key = obj["Key"]
            kb = obj["Size"] / 1024
            total_kb += kb
            modified = obj["LastModified"].strftime("%Y-%m-%d %H:%M")
            print(f"{key[:65]:<65} {kb:>8.1f} {modified}")

        if response.get("IsTruncated"):
            print(f"  ... (more files exist, showing first 50)")

        print(f"\nTotal: {len(objects)} files, {total_kb:.1f} KB")

        # Count by type
        pdfs = sum(1 for o in objects if o["Key"].endswith(".pdf"))
        htmls = sum(1 for o in objects if o["Key"].endswith(".html"))
        print(f"  PDFs: {pdfs}  |  HTML: {htmls}  |  Other: {len(objects)-pdfs-htmls}")

    except Exception as exc:
        print(f"  B2 check failed: {exc}")


def main():
    print(f"\n{'#'*70}")
    print("PHASE 1 PIPELINE VERIFICATION")
    print(f"{'#'*70}")

    session = get_session()
    try:
        check_companies(session)
        check_documents(session)
        check_facts(session)
    finally:
        session.close()

    check_b2()

    print(f"\n{SEP}")
    print("VERIFICATION COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()
