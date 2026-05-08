"""
Smoke Test V2 — 10 Tough Questions
Tests the 5 bug fix categories:
  1. SPOF routing (Q27 pattern) vs OEM dependency (Q28) vs capacity risk (Q29) vs misalignment (Q30)
  2. Tier synonym mapping ("Direct Manufacturer" → OEM)
  3. Industry group extraction + SQL routing
  4. Synthesis "not found" hallucination fix
  5. Compound filter (tier + role + employment range)
"""
import sys, time
sys.path.insert(0, ".")

from core_agent.agent_pipeline import EVAgent

QUESTIONS = [
    # ── T1: SPOF — should return direct list ✅ PASSING ──────────────────────
    {
        "id": "T1", "label": "SPOF",
        "question": "Which EV supply chain roles in Georgia are served by only a single company, representing single points of failure?",
        "expect": "direct bullet list of roles", "was_failing": False,
    },
    # ── T2: OEM dependency — FIX: no longer uses tier=OEM filter ─────────────
    {
        "id": "T2", "label": "OEM_DEPENDENCY",
        "question": "Which Battery Cell or Battery Pack suppliers in Georgia are sole-sourced by a single OEM, creating high dependency risk?",
        "expect": "Battery Cell/Pack companies (NOT tier=OEM filter)", "was_failing": True,
    },
    # ── T3: Capacity risk — FIX: synthesis must NOT re-filter ────────────────
    {
        "id": "T3", "label": "CAPACITY_RISK",
        "question": "Which Tier 1 and Tier 2 suppliers serving Hyundai in Georgia have fewer than 200 employees, suggesting limited production surge capacity?",
        "expect": "small Hyundai suppliers listed — not 'No matching'", "was_failing": True,
    },
    # ── T4: Misalignment ✅ PASSING ───────────────────────────────────────────
    {
        "id": "T4", "label": "MISALIGNMENT",
        "question": "Which Tier 2/3 companies classified as EV Relevant show supply chain misalignment between their role and tier classification?",
        "expect": "Tier 2/3 EV-relevant companies list", "was_failing": False,
    },
    # ── T5: Tier synonym ✅ PASSING ───────────────────────────────────────────
    {
        "id": "T5", "label": "TIER_SYNONYM",
        "question": "Which companies in Georgia are classified as Direct Manufacturers in the EV supply chain?",
        "expect": "OEM-tier companies (Kia, Hyundai, Rivian etc)", "was_failing": False,
    },
    # ── T6: Industry group ✅ PASSING ─────────────────────────────────────────
    {
        "id": "T6", "label": "INDUSTRY_GROUP",
        "question": "List all Tier 2 or Tier 3 companies in the Chemicals and Allied Products industry group in Georgia's EV supply chain.",
        "expect": "Archer Aviation (only company in Chemicals group)", "was_failing": False,
    },
    # ── T7: Synthesis trust ✅ PASSING ────────────────────────────────────────
    {
        "id": "T7", "label": "SYNTHESIS_TRUST",
        "question": "Which companies supply to Hyundai Motor Group in Georgia? List all of them.",
        "expect": "Hyundai supplier list — NOT 'database does not contain'", "was_failing": False,
    },
    # ── T8: Compound filter ✅ PASSING (minor: only listed 1 of 2) ───────────
    {
        "id": "T8", "label": "COMPOUND_FILTER",
        "question": "Which Tier 1 suppliers in Georgia have a Thermal Management role in the EV supply chain?",
        "expect": "Both Tier 1 Thermal Management companies listed", "was_failing": False,
    },
    # ── T9: Employment + OEM — FIX: OEM path now applies min_employment ───────
    {
        "id": "T9", "label": "EMP_RANGE",
        "question": "Which Georgia EV supply chain companies have more than 1000 employees and serve Kia or Rivian as primary OEMs?",
        "expect": "Only Rivian/Kia suppliers with >1000 emp (may be empty, that's OK)", "was_failing": True,
    },
    # ── T10: Facility type — FIX: facility_type now in query_companies ────────
    {
        "id": "T10", "label": "FACILITY_TIER",
        "question": "Which Tier 1 or Tier 2 companies in Georgia operate R&D facilities relevant to EV technology development?",
        "expect": "R&D facility companies, should list actual R&D companies not Manufacturing", "was_failing": True,
    },
]


def run():
    agent = EVAgent()
    results = []
    print("\n" + "=" * 70)
    print("  V2 SMOKE TEST — 10 Tough Questions")
    print("=" * 70)

    for q_info in QUESTIONS:
        print(f"\n{'─' * 70}")
        print(f"  [{q_info['id']}] {q_info['label']}")
        print(f"  Q: {q_info['question'][:90]}...")
        print(f"  Expect: {q_info['expect']}")
        print(f"{'─' * 70}")

        t0 = time.monotonic()
        result = agent.ask(q_info["question"])
        elapsed = time.monotonic() - t0

        entities = result["entities"]
        print(f"\n  ⚙  Entities: tier={entities.get('tier')} | oem={entities.get('oem')} | "
              f"industry={entities.get('industry_group')} | role={entities.get('ev_role') or entities.get('ev_role_list')}")
        print(f"  ⚙  Flags: risk={entities.get('is_risk_query')} | oem_dep={entities.get('is_oem_dependency')} | "
              f"capacity={entities.get('is_capacity_risk')} | misalign={entities.get('is_misalignment')}")
        print(f"  ⚙  Route: cypher={entities.get('cypher_used')} | rows={result.get('retrieved_count')}")
        print(f"\n  Answer preview: {result['answer'][:300]}")
        print(f"\n  ⏱  {elapsed:.1f}s")

        results.append({
            "id": q_info["id"],
            "label": q_info["label"],
            "elapsed": round(elapsed, 1),
            "rows": result.get("retrieved_count"),
            "entities": {k: v for k, v in entities.items() if v},
        })

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  [{r['id']}] {r['label']:<20} {r['elapsed']:>5.1f}s  rows={r['rows']}")
    avg = sum(r["elapsed"] for r in results) / len(results)
    print(f"\n  Avg: {avg:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    run()
