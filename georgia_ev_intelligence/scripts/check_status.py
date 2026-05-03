"""
Quick status check — what got saved to DB and B2 before the pipeline was stopped.
Run: venv\\Scripts\\python scripts\\check_status.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from shared.db import get_engine, get_session, verify_connection
from shared.storage import get_b2_client, verify_connection as b2_verify
from shared.config import Config

SEP = "=" * 65

def main():
    print(f"\n{'#'*65}")
    print("  PHASE 1 STATUS CHECK — What Got Saved")
    print(f"{'#'*65}\n")

    # DB check
    print(f"{'─'*65}")
    print("  DATABASE (Neon PostgreSQL)")
    print(f"{'─'*65}")

    if not verify_connection():
        print("  ❌ DB not reachable — check .env DATABASE_URL")
        return

    engine = get_engine()
    with engine.connect() as conn:

        # Companies
        total_cos = conn.execute(text("SELECT COUNT(*) FROM gev_companies")).scalar()
        print(f"\n  gev_companies table:  {total_cos} companies synced")

        # Documents summary
        doc_stats = conn.execute(text("""
            SELECT
                extraction_status,
                COUNT(*) as cnt,
                SUM(word_count) as total_words,
                SUM(file_size_bytes) as total_bytes
            FROM gev_documents
            GROUP BY extraction_status
        """)).fetchall()

        print(f"\n  gev_documents table:")
        total_docs = 0
        for r in doc_stats:
            kb = (r[3] or 0) / 1024
            print(f"    [{r[0] or 'unknown':10}] {r[1]:>4} docs  |  {(r[2] or 0):>7,} words  |  {kb:>7.1f} KB")
            total_docs += r[1]
        print(f"    {'TOTAL':>10}  {total_docs:>4} docs")

        # Per-company breakdown (only companies that have docs)
        company_docs = conn.execute(text("""
            SELECT
                company_name,
                COUNT(*) as doc_count,
                SUM(CASE WHEN extraction_status='extracted' THEN 1 ELSE 0 END) as ok,
                SUM(CASE WHEN b2_key IS NOT NULL THEN 1 ELSE 0 END) as in_b2,
                MAX(id) as last_doc_id
            FROM gev_documents
            GROUP BY company_name
            ORDER BY company_name
        """)).fetchall()

        print(f"\n  Per-company document breakdown:")
        print(f"  {'Company':<35} {'Docs':<6} {'OK':<5} {'In B2':<7} {'Last ID'}")
        print(f"  {'-'*35} {'-'*6} {'-'*5} {'-'*7} {'-'*8}")
        for r in company_docs:
            name = (r[0] or "")[:34]
            print(f"  {name:<35} {r[1]:<6} {r[2]:<5} {r[3]:<7} #{r[4]}")

        # Facts
        fact_count = conn.execute(text("SELECT COUNT(*) FROM gev_extracted_facts")).scalar()
        fact_cos   = conn.execute(text("SELECT COUNT(DISTINCT company_name) FROM gev_extracted_facts")).scalar()
        print(f"\n  gev_extracted_facts: {fact_count} facts across {fact_cos} companies")

        if fact_count > 0:
            top_facts = conn.execute(text("""
                SELECT company_name, fact_type, fact_value_text, confidence_score
                FROM gev_extracted_facts
                ORDER BY confidence_score DESC NULLS LAST
                LIMIT 8
            """)).fetchall()
            print(f"\n  Top facts extracted:")
            for f in top_facts:
                print(f"    [{f[0][:25]}] {f[1]:<20} {(f[2] or '')[:35]}  (conf={f[3] or 0:.2f})")

    # B2 check
    print(f"\n{'─'*65}")
    print("  BACKBLAZE B2 CLOUD STORAGE")
    print(f"{'─'*65}")

    try:
        cfg = Config.get()
        client = get_b2_client()
        resp = client.list_objects_v2(Bucket=cfg.b2_bucket, Prefix="documents/", MaxKeys=200)
        objects = resp.get("Contents", [])

        total_kb = sum(o["Size"] for o in objects) / 1024
        pdfs  = sum(1 for o in objects if o["Key"].endswith(".pdf"))
        htmls = sum(1 for o in objects if o["Key"].endswith(".html"))

        print(f"\n  Files in B2:   {len(objects)}  ({pdfs} PDFs, {htmls} HTML)")
        print(f"  Total size:    {total_kb:.1f} KB  ({total_kb/1024:.2f} MB)")

        # Unique companies in B2
        b2_companies = set()
        for o in objects:
            parts = o["Key"].split("/")
            if len(parts) >= 2:
                b2_companies.add(parts[1])
        print(f"  Companies in B2: {len(b2_companies)}")
        for co in sorted(b2_companies):
            co_files = [o for o in objects if f"/{co}/" in o["Key"]]
            print(f"    {co:<40} {len(co_files)} files")

        if resp.get("IsTruncated"):
            print("  (showing first 200 files — more exist)")

    except Exception as e:
        print(f"  ❌ B2 error: {e}")

    print(f"\n{'#'*65}")
    print("  STATUS CHECK COMPLETE")
    print(f"{'#'*65}\n")


if __name__ == "__main__":
    main()
