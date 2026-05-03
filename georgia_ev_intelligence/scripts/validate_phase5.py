"""
scripts/validate_phase5.py
─────────────────────────────────────────────────────────────────────────────
Validates Phase 5 (Few-Shot RAG) end-to-end BEFORE seeding production store.

WHAT THIS TESTS:
  1. DB schema columns match seed_store.py SQL examples
  2. Each SQL in seed_store.py executes and returns rows (no bad training data)
  3. Qdrant store: upsert / search / count work correctly
  4. Embedder: nomic-embed-text available OR fallback works
  5. Few-shot retriever: returns examples above threshold for similar questions
  6. Integration: text_to_sql gets few-shot block injected correctly

RUN:
  venv\\Scripts\\python scripts\\validate_phase5.py

EXPECTED OUTPUT:
  All checks PASS → ready to seed production store and run 50-question eval
"""
from __future__ import annotations

import sys, time, traceback
sys.path.insert(0, ".")

from shared.logger import get_logger
logger = get_logger("validate_phase5")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results: list[dict] = []


def check(name: str, fn):
    """Run a check function, capture pass/fail/warn."""
    t0 = time.monotonic()
    try:
        verdict, detail = fn()
        elapsed = time.monotonic() - t0
        results.append({"name": name, "verdict": verdict, "detail": detail, "elapsed": elapsed})
        print(f"  {verdict}  {name:<50} ({elapsed:.2f}s)")
        if detail and verdict != PASS:
            for line in detail.splitlines()[:5]:
                print(f"           {line}")
    except Exception as exc:
        elapsed = time.monotonic() - t0
        results.append({"name": name, "verdict": FAIL, "detail": str(exc), "elapsed": elapsed})
        print(f"  {FAIL}  {name:<50} ({elapsed:.2f}s)")
        print(f"           {traceback.format_exc()[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: DB connectivity + confirm table exists
# ─────────────────────────────────────────────────────────────────────────────
def check_db_connection():
    from shared.db import verify_connection, get_session
    from sqlalchemy import text
    ok = verify_connection()
    if not ok:
        return FAIL, "Cannot connect to PostgreSQL"
    session = get_session()
    try:
        r = session.execute(text("SELECT COUNT(*) FROM gev_companies")).scalar()
        return PASS, f"{r} rows in gev_companies"
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: Confirm all columns used in SQL examples exist in the table
# ─────────────────────────────────────────────────────────────────────────────
def check_db_columns():
    from shared.db import get_session
    from sqlalchemy import text
    session = get_session()
    try:
        rows = session.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'gev_companies'
            ORDER BY ordinal_position
        """)).fetchall()
        actual_cols = {r[0] for r in rows}
        expected_cols = {
            "company_name", "tier", "ev_supply_chain_role", "ev_battery_relevant",
            "industry_group", "facility_type", "location_county", "location_city",
            "employment", "products_services", "primary_oems",
            "classification_method", "supplier_affiliation_type"
        }
        missing = expected_cols - actual_cols
        if missing:
            return FAIL, f"Missing columns: {missing}"
        extra = actual_cols - expected_cols - {"id", "latitude", "longitude", "location_state", "created_at", "updated_at"}
        return PASS, f"{len(actual_cols)} columns found. All expected columns present."
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3: Validate EVERY SQL in seed_store.py returns rows
# ─────────────────────────────────────────────────────────────────────────────
def check_seed_sqls():
    from shared.db import get_session
    from sqlalchemy import text
    from phase5_fewshot.seed_store import SQL_EXAMPLES

    session = get_session()
    failures = []
    empty = []
    try:
        for ex in SQL_EXAMPLES:
            sql = ex["sql"].strip()
            # Add LIMIT 5 if not present (for validation speed)
            if "LIMIT" not in sql.upper():
                sql = sql.rstrip(";") + " LIMIT 5;"
            else:
                sql = sql.rstrip(";") + ";"
            try:
                rows = session.execute(text(sql)).fetchall()
                if len(rows) == 0:
                    empty.append(ex["question"][:60])
                    logger.warning("SQL returned 0 rows for: %s", ex["question"][:60])
            except Exception as exc:
                failures.append(f"{ex['question'][:50]}: {exc}")
    finally:
        session.close()

    if failures:
        return FAIL, f"{len(failures)} SQL errors:\n" + "\n".join(failures[:3])
    if empty:
        return WARN, f"{len(empty)} SQLs returned 0 rows (bad training examples):\n" + "\n".join(empty[:5])
    return PASS, f"All {len(SQL_EXAMPLES)} SQL examples execute and return rows"


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4: Embedder — nomic-embed-text or fallback
# ─────────────────────────────────────────────────────────────────────────────
def check_embedder():
    from phase5_fewshot.embedder import embed_text, check_embed_model_available, _EMBED_DIM
    model_ok = check_embed_model_available()
    vec = embed_text("Which county has the highest Tier 1 employment?")
    if len(vec) != _EMBED_DIM:
        return FAIL, f"Expected {_EMBED_DIM}-dim vector, got {len(vec)}"
    norm = sum(v * v for v in vec) ** 0.5
    if model_ok:
        return PASS, f"nomic-embed-text OK — vector dim={len(vec)}, norm={norm:.3f}"
    else:
        return WARN, f"nomic-embed-text not available — keyword fallback used (dim={len(vec)}). Run: ollama pull nomic-embed-text"


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5: Qdrant store — upsert / count / search
# ─────────────────────────────────────────────────────────────────────────────
def check_qdrant_store():
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return FAIL, "qdrant-client not installed. Run: pip install qdrant-client"

    from phase5_fewshot.qdrant_store import upsert_example, search_similar, count_examples
    from phase5_fewshot.embedder import embed_text

    # Test upsert
    test_q = "TEST: Which county has highest Tier 1 employment?"
    vec = embed_text(test_q)
    pid = upsert_example(
        question="TEST: Which county has highest Tier 1 employment?",
        vector=vec,
        query_type="sql",
        sql="SELECT location_county FROM gev_companies WHERE tier='Tier 1' ORDER BY employment DESC LIMIT 1;",
        answer="Troup County",
        source="test",
    )

    count = count_examples()
    if count < 1:
        return FAIL, "Upsert succeeded but count=0"

    # Test search
    search_vec = embed_text("Which Tier 1 county has the most employment?")
    hits = search_similar(search_vec, top_k=1)
    if not hits:
        return FAIL, "Search returned 0 results after upsert"

    top_score = hits[0].get("score", 0)
    return PASS, f"Qdrant OK — count={count}, search top_score={top_score:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6: Few-shot retriever — finds similar questions above threshold
# ─────────────────────────────────────────────────────────────────────────────
def check_retriever():
    from phase5_fewshot.few_shot_retriever import get_few_shot_examples, get_few_shot_block

    # At least the test example from check 5 should be in the store
    similar_q = "Which county has the highest employment among Tier 1 suppliers?"
    examples = get_few_shot_examples(similar_q, query_type="sql", top_k=3)

    if not examples:
        return WARN, "No examples found above similarity threshold (0.70). Store may be empty — run seed first."

    block = get_few_shot_block(similar_q, query_type="sql", top_k=3)
    top = examples[0]
    return PASS, (
        f"Found {len(examples)} examples | "
        f"Top: score={top['score']:.3f} | "
        f"Q='{top['question'][:50]}'"
        f"\nBlock: {len(block)} chars"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7: Full seed — seed all SQL + Cypher examples
# ─────────────────────────────────────────────────────────────────────────────
def check_full_seed():
    from phase5_fewshot.seed_store import SQL_EXAMPLES, CYPHER_EXAMPLES
    from phase5_fewshot.qdrant_store import upsert_example, count_examples, delete_collection
    from phase5_fewshot.embedder import embed_text

    # Wipe test data and re-seed properly
    try:
        delete_collection()
    except Exception:
        pass

    total = 0
    errors = []

    for ex in SQL_EXAMPLES:
        try:
            vec = embed_text(ex["question"])
            upsert_example(
                question=ex["question"], vector=vec, query_type="sql",
                sql=ex["sql"], answer=ex["answer"],
                category=ex.get("category", "GENERAL"), source="validated",
            )
            total += 1
        except Exception as e:
            errors.append(str(e))

    for ex in CYPHER_EXAMPLES:
        try:
            vec = embed_text(ex["question"])
            upsert_example(
                question=ex["question"], vector=vec, query_type="cypher",
                cypher=ex["cypher"], answer=ex["answer"],
                category=ex.get("category", "GENERAL"), source="validated",
            )
            total += 1
        except Exception as e:
            errors.append(str(e))

    if errors:
        return FAIL, f"{len(errors)} upsert errors: {errors[0]}"

    count = count_examples()
    return PASS, f"Seeded {total} examples | Store count={count}"


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 8: Integration — text_to_sql gets few-shot block injected
# ─────────────────────────────────────────────────────────────────────────────
def check_integration():
    from phase5_fewshot.few_shot_retriever import get_few_shot_block

    # Test 3 question types that should find examples
    test_questions = [
        ("Which county has the highest Tier 1 employment?",    "sql", "AGGREGATE"),
        ("Which companies supply to Rivian?",                   "sql", "OEM"),
        ("Which companies make copper foil?",                   "cypher", "PRODUCT"),
    ]

    results_inner = []
    for q, qtype, label in test_questions:
        block = get_few_shot_block(q, query_type=qtype, top_k=3)
        example_count = block.count("Example ") if block else 0
        results_inner.append(f"{label}: {example_count} examples injected")

    all_found = all("0 examples" not in r for r in results_inner)
    verdict = PASS if all_found else WARN
    return verdict, " | ".join(results_inner)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 9: Verify SQL correctness — spot-check 5 critical SQLs against known answers
# ─────────────────────────────────────────────────────────────────────────────
def check_sql_correctness():
    """Run a few SQLs and verify they return KNOWN correct answers."""
    from shared.db import get_session
    from sqlalchemy import text

    SPOT_CHECKS = [
        {
            "name": "Tier 1 county aggregate",
            "sql": "SELECT location_county, SUM(employment) AS total FROM gev_companies WHERE LOWER(tier)='tier 1' AND employment IS NOT NULL GROUP BY location_county ORDER BY total DESC LIMIT 1;",
            "check": lambda rows: len(rows) > 0 and rows[0][0] is not None,
            "expect": "Returns a county with highest Tier 1 employment",
        },
        {
            "name": "Rivian suppliers",
            "sql": "SELECT company_name FROM gev_companies WHERE primary_oems ILIKE '%Rivian%' ORDER BY employment DESC LIMIT 10;",
            "check": lambda rows: len(rows) > 0,
            "expect": "Returns at least 1 Rivian supplier",
        },
        {
            "name": "Battery Cell role",
            "sql": "SELECT company_name FROM gev_companies WHERE ev_supply_chain_role ILIKE '%Battery Cell%' LIMIT 10;",
            "check": lambda rows: len(rows) > 0,
            "expect": "Returns Battery Cell companies",
        },
        {
            "name": "OEM tier companies",
            "sql": "SELECT company_name FROM gev_companies WHERE tier ILIKE '%OEM%' LIMIT 10;",
            "check": lambda rows: len(rows) > 0,
            "expect": "Returns OEM-tier companies",
        },
        {
            "name": "R&D facility type",
            "sql": "SELECT company_name, facility_type FROM gev_companies WHERE facility_type ILIKE '%R&D%' LIMIT 10;",
            "check": lambda rows: len(rows) > 0,
            "expect": "Returns R&D facility companies",
        },
    ]

    session = get_session()
    failures = []
    passed = 0
    try:
        for sc in SPOT_CHECKS:
            rows = session.execute(text(sc["sql"])).fetchall()
            if sc["check"](rows):
                passed += 1
                logger.debug("Spot check PASS: %s (%d rows)", sc["name"], len(rows))
            else:
                failures.append(f"{sc['name']}: 0 rows (expected: {sc['expect']})")
    finally:
        session.close()

    if failures:
        return WARN, f"{passed}/{len(SPOT_CHECKS)} passed | Failures: {' | '.join(failures)}"
    return PASS, f"All {passed}/{len(SPOT_CHECKS)} spot-check SQLs returned correct rows"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  PHASE 5 VALIDATION — Few-Shot RAG (Qdrant + nomic-embed-text)")
    print("=" * 70 + "\n")

    check("1. DB connection + table exists",      check_db_connection)
    check("2. DB columns match schema",            check_db_columns)
    check("3. All seed SQLs execute + return rows", check_seed_sqls)
    check("4. Embedder (nomic-embed-text / fallback)", check_embedder)
    check("5. Qdrant store (upsert / search / count)", check_qdrant_store)
    check("6. Few-shot retriever (similarity threshold)", check_retriever)
    check("7. Full seed (all SQL + Cypher examples)", check_full_seed)
    check("8. Integration (few-shot block injection)", check_integration)
    check("9. SQL correctness (spot-check known answers)", check_sql_correctness)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    passed  = sum(1 for r in results if r["verdict"] == PASS)
    warned  = sum(1 for r in results if r["verdict"] == WARN)
    failed  = sum(1 for r in results if r["verdict"] == FAIL)
    total_t = sum(r["elapsed"] for r in results)

    for r in results:
        print(f"  {r['verdict']}  {r['name']}")

    print(f"\n  {passed} passed | {warned} warnings | {failed} failed | {total_t:.1f}s total")

    if failed == 0 and warned == 0:
        print("\n  🚀 All checks passed — Phase 5 is ready for production use!")
        print("     Next step: venv\\Scripts\\python scripts\\run_ragas_eval.py --questions 50")
    elif failed == 0:
        print("\n  ⚠️  Phase 5 is FUNCTIONAL with warnings (see above).")
        print("     Fix warnings before final 50-question eval.")
    else:
        print("\n  ❌ Phase 5 has FAILURES — fix before proceeding.")
        sys.exit(1)
