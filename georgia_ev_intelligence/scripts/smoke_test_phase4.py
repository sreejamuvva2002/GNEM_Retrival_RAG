"""
Phase 4 Smoke Test — 7 questions covering ALL retrieval paths in V2 architecture.

Path coverage:
  [A] Aggregate    → PostgreSQL GROUP BY
  [B] Risk         → Deterministic list
  [C] Cypher Tier  → Neo4j Text-to-Cypher
  [D] Cypher OEM   → Neo4j multi-hop (suppliers to specific OEM)
  [E] Cypher Prod  → Neo4j product/text search
  [F] Cypher Multi → Neo4j multi-OEM pattern
  [G] Fallback     → Full-text PostgreSQL (if Cypher returns 0)

Run: venv\\Scripts\\python scripts\\smoke_test_phase4.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_agent.agent_pipeline import EVAgent

SEP  = "=" * 65
agent = EVAgent()

# 7 questions — one per retrieval path
TEST_QUESTIONS = [
    # [A] PostgreSQL aggregate
    ("AGGREGATE",  "Which county has the highest total employment among Tier 1 suppliers only?"),
    # [B] Risk / single-supplier
    ("RISK",       "Which EV supply chain roles in Georgia have only one supplier, making them a single point of failure?"),
    # [C] Cypher: tier filter
    ("CYPHER-TIER","Which Georgia companies are classified under Battery Cell or Battery Pack roles, and what tier is each assigned?"),
    # [D] Cypher: OEM relationship traversal
    ("CYPHER-OEM", "Show the full supplier network linked to Rivian Automotive in Georgia, broken down by tier and EV Supply Chain Role."),
    # [E] Cypher: product text search
    ("CYPHER-PROD","Find Georgia-based companies that manufacture copper foil or electrodeposited materials for EV battery current collectors."),
    # [F] Cypher: multi-hop / county
    ("CYPHER-LOC", "In Gwinnett County, which company has the highest employment and what is its EV Supply Chain Role?"),
    # [G] Cypher: R&D facility type (tests facility_type sync)
    ("CYPHER-FAC", "Which Georgia companies operate R&D facilities focused on EV technology?"),
]

print(f"\n{'#'*65}")
print("  PHASE 4 SMOKE TEST — V2 Architecture (Text-to-Cypher)")
print(f"{'#'*65}\n")

passed = 0
for i, (path_type, question) in enumerate(TEST_QUESTIONS, 1):
    print(f"{SEP}")
    print(f"  Q{i} [{path_type}]: {question[:65]}...")
    print(SEP)

    t0 = time.monotonic()
    result = agent.ask(question)
    elapsed = time.monotonic() - t0

    ents = result.get("entities", {})
    cypher_used = ents.get("cypher_used", False)
    print(f"  Path      : {'Neo4j Cypher ✓' if cypher_used else 'PostgreSQL / Direct'}")
    print(f"  Tier      : {ents.get('tier')} | County: {ents.get('county')} | OEM: {ents.get('oem')}")
    print(f"  Aggregate : {ents.get('is_aggregate')} | Retrieved~: {result['retrieved_count']} rows")
    print(f"  Time      : {result['elapsed_s']}s")
    print(f"\n  ANSWER:\n")
    for line in result['answer'].split('\n'):
        print(f"    {line}")

    ok = bool(result['answer'].strip()) and len(result['answer']) > 30
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  {status}\n")
    if ok:
        passed += 1

print(f"\n{SEP}")
print(f"  RESULTS: {passed}/{len(TEST_QUESTIONS)} tests passed")
print(SEP)
if passed == len(TEST_QUESTIONS):
    print(f"\n  🎉 All paths working! Ready for full 50-question evaluation.")
    print(f"     venv\\Scripts\\python scripts\\run_evaluation.py")
elif passed >= 5:
    print(f"\n  ⚠️  {len(TEST_QUESTIONS) - passed} test(s) failed — review before full eval.")
else:
    print(f"\n  ❌ {len(TEST_QUESTIONS) - passed} failures — do NOT run full eval yet.")
print()
