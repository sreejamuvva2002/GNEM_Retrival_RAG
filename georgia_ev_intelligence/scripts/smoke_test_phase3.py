"""
Phase 3 Smoke Test — Neo4j Graph
5 tests to verify the graph is correct before building the agent.

Run: venv\\Scripts\\python scripts\\smoke_test_phase3.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_storage.graph_loader import verify_connection, get_graph_stats, get_driver
from shared.logger import get_logger

SEP = "=" * 55
logger = get_logger("smoke.phase3")
results = []

def run_test(name, fn):
    print(f"\n  Testing: {name}")
    try:
        fn()
        print(f"  ✅ PASS: {name}")
        results.append((name, True, None))
    except AssertionError as e:
        print(f"  ❌ FAIL: {name}\n         → {e}")
        results.append((name, False, str(e)))
    except Exception as e:
        print(f"  ❌ FAIL: {name}\n         → {e}")
        results.append((name, False, str(e)))


def test_connection():
    result = verify_connection()
    assert result["ok"], f"Neo4j unreachable: {result.get('error')}"
    print(f"         → Connected to AuraDB")


def test_company_nodes():
    stats = get_graph_stats()
    count = stats.get("nodes_company", 0)
    assert count >= 100, f"Too few Company nodes: {count} (expected ~193)"
    print(f"         → {count} Company nodes")


def test_relationships_exist():
    stats = get_graph_stats()
    total_rels = stats.get("total_rels", 0)
    assert total_rels >= 200, f"Too few relationships: {total_rels} (expected ~3500)"
    print(f"         → {total_rels} total relationships")


def test_cypher_tier_query():
    """Can we find Tier 1 companies in a specific county?"""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Company)
            WHERE c.tier = 'Tier 1'
            RETURN count(c) AS cnt
        """)
        count = result.single()["cnt"]
    assert count >= 5, f"Expected >= 5 Tier 1 companies, got {count}"
    print(f"         → {count} Tier 1 companies found via Cypher")


def test_cypher_oem_suppliers():
    """Can we traverse Company → OEM relationships?"""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Company)-[:SUPPLIES_TO]->(o:OEM)
            RETURN o.name AS oem, count(c) AS supplier_count
            ORDER BY supplier_count DESC
            LIMIT 5
        """)
        rows = result.data()
    assert len(rows) >= 1, "No SUPPLIES_TO relationships found"
    top = rows[0]
    print(f"         → Top OEM: {top['oem']} ({top['supplier_count']} suppliers)")
    for r in rows:
        print(f"           {r['oem']}: {r['supplier_count']} companies")


def main():
    print(f"\n{'#'*55}")
    print("  PHASE 3 SMOKE TEST — Neo4j Knowledge Graph")
    print(f"{'#'*55}")

    run_test("1. Neo4j AuraDB connection",         test_connection)
    run_test("2. Company nodes loaded (~193)",      test_company_nodes)
    run_test("3. Relationships exist (~3500)",      test_relationships_exist)
    run_test("4. Cypher: Tier 1 filter query",      test_cypher_tier_query)
    run_test("5. Cypher: OEM supplier traversal",   test_cypher_oem_suppliers)

    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print(f"\n{SEP}")
    print(f"  RESULTS: {passed}/{total} tests passed")
    print(SEP)
    for name, ok, err in results:
        print(f"  {'✅' if ok else '❌'} {name}")
        if err and not ok:
            print(f"       {err[:100]}")

    if passed == total:
        print(f"\n  🎉 All graph tests passed — ready for Phase 4 (Agent)!")
    else:
        print(f"\n  ⚠️  Fix failures before building the agent")
    print()

if __name__ == "__main__":
    main()
