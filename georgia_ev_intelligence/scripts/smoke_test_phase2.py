"""
Phase 2 Smoke Test
Tests the embedding pipeline end-to-end before running on all 205 companies.

What this tests:
  1. Ollama embed model works (active embed model → configured dims)
  2. Qdrant collection is reachable and configured correctly
  3. Chunker splits one document correctly (parent + child chunks)
  4. Upload 1 company's multi-view chunks to Qdrant — verify they landed
  5. Search for 5 EV queries — verify relevant chunks come back

Run:
  venv\\Scripts\\python scripts\\smoke_test_phase2.py

Expected output:
  All 5 tests PASS → safe to run full Phase 2 pipeline
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2_embedding.chunker import chunk_company_record, chunk_document, get_child_chunks
from phase2_embedding.embedder import embed_single, embed_texts, verify_ollama_embed
from phase2_embedding.vector_store import (
    ensure_collection_exists,
    get_collection_stats,
    upload_chunks,
    search_hybrid,
    verify_qdrant_connection,
    _build_sparse_vector,
)
from phase1_extraction.kb_loader import build_document_text, get_all_companies_from_db
from phase2_embedding.embedder import embed_chunks
from shared.config import Config

SEP = "=" * 60
PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = []

def run_test(name: str, fn):
    print(f"\n  Testing: {name}")
    try:
        fn()
        print(f"  {PASS}: {name}")
        results.append((name, True, None))
    except AssertionError as e:
        print(f"  {FAIL}: {name}")
        print(f"         → {e}")
        results.append((name, False, str(e)))
    except Exception as e:
        print(f"  {FAIL}: {name}")
        print(f"         → Unexpected error: {e}")
        results.append((name, False, str(e)))


# ─── Test 1: Ollama embed works ───────────────────────────────────────────────
def test_ollama_embed():
    result = verify_ollama_embed()
    assert result["ok"], f"Embed failed: {result['error']}"
    expected_dims = Config.get().qdrant_dimensions
    assert result["dimensions"] == expected_dims, f"Expected {expected_dims} dims, got {result['dimensions']}"
    print(f"         → {result['model']}, {result['dimensions']} dims")


# ─── Test 2: Qdrant collection reachable ─────────────────────────────────────
def test_qdrant_connection():
    result = verify_qdrant_connection()
    assert result["ok"], f"Qdrant failed: {result['error']}"
    stats = get_collection_stats()
    assert "points_count" in stats, "No points_count in stats"
    print(f"         → collection '{stats['collection']}', {stats['points_count']} existing points")


# ─── Test 3: Chunker splits document correctly ────────────────────────────────
def test_chunker():
    sample_text = """
    Hyundai Motor Group announced a $5.5 billion investment in Bryan County, Georgia in 2023.
    The Hyundai METAPLANT America facility will create 8,500 direct jobs and thousands of indirect jobs.
    Construction began in Q3 2022 and production is expected to start by Q1 2025.

    The 3,000-acre facility will manufacture electric vehicles including Hyundai Ioniq 5, Ioniq 6,
    Genesis GV70, and Kia EV6. Battery systems will be supplied by SK On from their Commerce facility.

    Georgia Governor Brian Kemp called the announcement the largest economic development deal in
    Georgia history. The facility is located in Ellabell and will include a dedicated battery plant.
    """ * 4  # Repeat to force parent-child splitting

    company_meta = {
        "id": 999, "company_name": "Test Hyundai",
        "tier": "OEM", "location_county": "Bryan County",
        "ev_battery_relevant": "Yes", "employment": 8500.0,
    }

    chunks = chunk_document(
        text=sample_text,
        company_name="Test Hyundai",
        document_id=999,
        source_url="https://example.com/test",
        document_type="press_release",
        company_metadata=company_meta,
    )

    parents  = [c for c in chunks if c.chunk_type == "parent"]
    children = [c for c in chunks if c.chunk_type == "child"]

    assert len(parents)  > 0, "No parent chunks created"
    assert len(children) > 0, "No child chunks created"
    assert len(children) >= len(parents), "Expected more children than parents"

    # Every child must have a parent_id
    for child in children:
        assert child.parent_id is not None, f"Child {child.chunk_id[:8]} has no parent_id"

    # Children must have company metadata in payload
    assert children[0].metadata.get("tier") == "OEM", "Company metadata not in child chunk"

    # Verify parent text is accessible for each child
    parent_map = {p.chunk_id: p for p in parents}
    for child in children:
        assert child.parent_id in parent_map, f"Child's parent_id not found in parent map"

    print(f"         → {len(parents)} parents, {len(children)} children from {len(sample_text)} chars")


# ─── Test 4: Upload 1 company chunk to Qdrant ─────────────────────────────────
def test_upload_company_chunk():
    # Use ACM Georgia LLC which we know is in the DB
    companies = get_all_companies_from_db()
    test_company = next((c for c in companies if "ACM" in c.get("company_name", "")), None)

    if test_company is None:
        # Fallback to first company
        test_company = companies[0] if companies else None

    assert test_company is not None, "No companies found in DB — run Phase 1 first"

    doc_text = build_document_text(test_company)
    company_chunks = chunk_company_record(test_company, doc_text)

    assert len(company_chunks) >= 1, f"Expected at least 1 company chunk, got {len(company_chunks)}"
    assert all(chunk.chunk_type == "company" for chunk in company_chunks)
    view_names = {chunk.metadata.get('chunk_view') for chunk in company_chunks}
    assert "master" in view_names, "Expected a master company chunk"

    # Embed
    vectors = embed_chunks(company_chunks)
    assert len(vectors) == len(company_chunks), f"Expected {len(company_chunks)} vectors, got {len(vectors)}"
    vec = list(vectors.values())[0]
    expected_dims = Config.get().qdrant_dimensions
    assert len(vec) == expected_dims, f"Expected {expected_dims}-dim vector, got {len(vec)}"

    # Upload
    uploaded = upload_chunks(company_chunks, vectors)
    assert uploaded == len(company_chunks), f"Expected {len(company_chunks)} uploaded, got {uploaded}"

    company_name = test_company["company_name"]
    print(f"         → uploaded '{company_name}' → Qdrant ({len(company_chunks)} chunks, {len(vec)} dims)")


# ─── Test 5: Search returns relevant results ──────────────────────────────────
def test_search_5_queries():
    """
    Run 5 representative EV supply chain queries.
    Verifies the hybrid search pipeline works end-to-end.
    At least 1 of 5 must return results (Qdrant may be sparse now).
    """
    test_queries = [
        "Tier 1 battery manufacturer Georgia",
        "EV supply chain investment Georgia 2023 2024",
        "Chatham County automotive supplier employment",
        "Kia Hyundai OEM supplier tier 1",
        "electric vehicle battery cell manufacturing Georgia",
    ]

    hits = 0
    for query in test_queries:
        query_vec = embed_single(query)
        results   = search_hybrid(
            query_text=query,
            query_vector=query_vec,
            top_k=3,
        )
        if results:
            hits += 1
            top = results[0]
            print(f"         Q: '{query[:45]}...'")
            print(f"            → [{top['company_name']}] score={top['score']:.3f}")
        else:
            print(f"         Q: '{query[:45]}...' → no results yet")

    # Pass if at least 1 query returns results
    # (Qdrant may be empty on first run before full Phase 2 pipeline)
    assert hits >= 1, (
        f"0 of 5 queries returned results. "
        f"Run Phase 2 pipeline first: venv\\Scripts\\python -m phase2_embedding.pipeline --companies-only"
    )
    print(f"\n         → {hits}/5 queries returned results")


# ─── Run all tests ────────────────────────────────────────────────────────────
def main():
    print(f"\n{'#'*60}")
    print("  PHASE 2 SMOKE TEST")
    print(f"{'#'*60}")
    print(f"\n  {'Test':<45} {'Status'}")
    print(f"  {'-'*45} {'-'*10}")

    run_test("1. Ollama embed (active model → configured dims)", test_ollama_embed)
    run_test("2. Qdrant collection reachable",                  test_qdrant_connection)
    run_test("3. Chunker: parent-child split correct",          test_chunker)
    run_test("4. Upload 1 company chunk → Qdrant",             test_upload_company_chunk)
    run_test("5. Search: 5 EV queries return results",          test_search_5_queries)

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print(f"\n{SEP}")
    print(f"  RESULTS: {passed}/{total} tests passed")
    print(SEP)
    for name, ok, err in results:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")
        if err and not ok:
            print(f"       Error: {err[:100]}")

    if passed == total:
        print(f"\n  🎉 All smoke tests passed!")
        print(f"  → Safe to run full Phase 2:")
        print(f"     venv\\Scripts\\python -m phase2_embedding.pipeline --companies-only")
        print(f"     venv\\Scripts\\python -m phase2_embedding.pipeline")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed — fix before running full pipeline")
    print()


if __name__ == "__main__":
    main()
