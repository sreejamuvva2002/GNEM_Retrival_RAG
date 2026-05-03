"""
Phase 4 — Full Evaluation Runner
Runs all 50 curated GNEM questions through the agent pipeline.
Saves detailed results to scripts/evaluation_results.md for review.

Run: venv\Scripts\python scripts\run_evaluation.py
"""
import sys, time, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase4_agent.pipeline import EVAgent

# ── All 50 evaluation questions ───────────────────────────────────────────────

QUESTIONS = [
    # 1  Supply Chain Mapping & Visibility
    "Show all Tier 1/2 suppliers in Georgia, list their EV Supply Chain Role and Product / Service.",
    # 2
    "Which Georgia companies are classified under Battery Cell or Battery Pack roles, and what tier is each assigned?",
    # 3
    "Map all Thermal Management suppliers in Georgia and show which Primary OEMs they are linked to.",
    # 4
    "List every Georgia company classified under Power Electronics or Charging Infrastructure, along with their employment and county.",
    # 5
    "Which companies are classified as Direct Manufacturer, and what EV Supply Chain Roles do they cover?",
    # 6
    "What locations does Novelis Inc. operate in, and what primary facility types are associated with each location?",
    # 7
    "In Gwinnett County, which company has the highest Employment and what is its EV Supply Chain Role?",
    # 8
    "Which county have the highest total Employment among Tier 1 suppliers only?",
    # 9
    "Which county has the highest total employment across all companies, and what is the combined employment?",
    # 10
    "Show all Vehicle Assembly OEMs in Georgia and the full set of Tier 1 suppliers connected to each with their EV Supply Chain Role.",
    # 11
    "Identify the primary products and services associated with Sewon America Inc. across its different operations.",
    # 12
    "List all Tier 2/3 companies in Georgia with primary involvement in the electric vehicle or battery supply chain.",
    # 13
    "Show the full supplier network linked to Rivian Automotive in Georgia, broken down by tier and EV Supply Chain Role.",
    # 14
    "Which Georgia companies produce battery materials such as anodes, cathodes, electrolytes, or copper foil?",
    # 15
    "Identify all Georgia companies with an EV Supply Chain Role related to wiring harnesses and show their employment.",
    # 16  Supplier Discovery & Matchmaking
    "Find Georgia-based Tier 1 or Tier 1/2 suppliers capable of producing battery electrolytes or lithium-ion battery materials.",
    # 17
    "Which Georgia companies manufacture high-voltage wiring harnesses or EV electrical distribution components suitable for BEV platforms?",
    # 18
    "Identify Georgia Tier 2/3 companies in the Electronic and Electrical Equipment industry group that could support EV power electronics.",
    # 19
    "Find Georgia-based companies that manufacture copper foil or electrodeposited materials suitable for EV battery current collectors.",
    # 20
    "Which Georgia Tier 1/2 companies produce engineered plastics, polymers, or composite materials applicable to EV structural components?",
    # 21
    "Find Georgia suppliers with existing Hyundai Kia contracts that could be expanded to support Hyundai Metaplant's EV battery production ramp-up.",
    # 22
    "Identify Georgia companies producing DC-to-DC converters, capacitors, or power electronics components relevant to EV drivetrains.",
    # 23
    "Which Georgia companies provide powder coating-related products or services, and what tier are they classified under?",
    # 24
    "Which Georgia companies manufacture battery parts or enclosure systems and are classified as Tier 1/2?",
    # 25
    "Find Tier 2/3 Georgia-based suppliers with employment over 300 that are classified as General Automotive.",
    # 26
    "Identify Georgia Tier 2/3 companies in the Chemicals and Allied Products industry group and list their products.",
    # 27  Supply Chain Risk & Resilience
    "Which EV Supply Chain Roles in Georgia are served by only a single company, creating a single-point-of-failure risk?",
    # 28
    "Which Georgia Battery Cell or Battery Pack suppliers are sole-sourced by a specific OEM, indicating high dependency risk?",
    # 29
    "For Hyundai Metaplant, how many of its Georgia-based EV component suppliers have fewer than 200 employees?",
    # 30
    "Identify Georgia Tier 2/3 suppliers that are EV Relevant, classified as General Automotive, and provide materials to Battery Cell or Battery Pack companies.",
    # 31
    "Identify all Georgia-based Tier 1/2 automotive suppliers that maintain a diversified customer base serving multiple OEMs.",
    # 32
    "Identify Georgia companies in the Thermal Management or Power Electronics role with fewer than 200 employees.",
    # 33
    "Identify any EV-relevant Georgia companies classified as OEM Footprint or OEM Supply Chain?",
    # 34
    "Top 10 Georgia companies based on employment size that supply both General Automotive and EV-specific roles.",
    # 35  Product & Technology Trends
    "How many Georgia companies are now producing lithium-ion battery materials, cells, or electrolytes?",
    # 36
    "Which Georgia Tier 2/3 suppliers currently produce lightweight aluminum or composite materials and are growing their EV-specific customer base?",
    # 37
    "Identify Georgia companies whose product descriptions include high-voltage, DC-to-DC, inverter, or motor controller product signals.",
    # 38
    "Which Georgia automotive companies employ over 1,000 workers but are currently categorized as only Indirectly Relevant to EVs?",
    # 39
    "Which three Georgia companies have the largest employment in EV thermal management, and how many employees does each have?",
    # 40
    "Which companies are involved in thermal-related products or services, and what roles and facility types do they have?",
    # 41
    "Which Tier 1/2 Georgia companies listed under General Automotive suggest a gradual evolution toward EV-related products or markets?",
    # 42
    "How is demand for thermal management solutions reflected in the number and employment size of Georgia suppliers?",
    # 43
    "Which Georgia companies are involved in battery recycling or second-life battery processing?",
    # 44
    "Which Georgia suppliers appear to play innovation-stage roles through research, development, or prototyping activity?",
    # 45
    "Which Georgia suppliers currently serving traditional OEMs are also linked to EV-native OEMs, showing dual-platform capability?",
    # 46  Site Selection & Expansion Planning
    "Identify Georgia areas that currently lack Battery Cell or Battery Pack suppliers but have existing Tier 1 automotive presence.",
    # 47
    "For a new Tier 1 battery thermal management company looking to locate in Georgia, which areas offer the highest concentration of existing materials suppliers?",
    # 48
    "How many Georgia areas have concentrated Manufacturing Plant facilities but no EV-specific production yet?",
    # 49
    "For an international battery materials company seeking a Georgia location, which areas have existing Chemicals and Allied Products infrastructure?",
    # 50
    "Which Georgia areas have R&D facility types in the automotive sector, suggesting innovation infrastructure?",
]

assert len(QUESTIONS) == 50, f"Expected 50 questions, got {len(QUESTIONS)}"


# ── Run evaluation ────────────────────────────────────────────────────────────

def run_evaluation():
    agent = EVAgent()
    results = []
    total = len(QUESTIONS)
    start_total = time.monotonic()

    print(f"\n{'#'*65}")
    print(f"  PHASE 4 FULL EVALUATION — {total} Questions")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*65}\n")

    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i:02d}/{total}] {question[:80]}...")
        result = agent.ask(question)
        results.append(result)

        ents = result.get("entities", {})
        answer_preview = result["answer"][:120].replace("\n", " ")
        print(f"         ⏱ {result['elapsed_s']}s | retrieved={result['retrieved_count']} | "
              f"tier={ents.get('tier')} county={ents.get('county')} "
              f"company={ents.get('company')} oem={ents.get('oem')}")
        print(f"         → {answer_preview}...")
        print()

    elapsed_total = time.monotonic() - start_total
    print(f"\nTotal time: {elapsed_total/60:.1f} minutes\n")
    return results


# ── Save results to markdown ──────────────────────────────────────────────────

def save_markdown(results: list[dict]) -> str:
    out_path = Path(__file__).parent / "evaluation_results.md"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Phase 4 — Full Evaluation Results",
        f"**Run Date:** {ts}",
        f"**Total Questions:** {len(results)}",
        f"**Total Time:** {sum(r['elapsed_s'] for r in results):.1f}s "
        f"({sum(r['elapsed_s'] for r in results)/60:.1f} min)",
        "",
        "---",
        "",
    ]

    # Summary table
    lines += [
        "## Summary Table",
        "",
        "| # | Question (truncated) | Retrieved | Time | Answer (preview) |",
        "|---|---------------------|-----------|------|-----------------|",
    ]
    for i, r in enumerate(results, 1):
        q = r["question"][:60].replace("|", "/")
        ans = r["answer"][:80].replace("\n", " ").replace("|", "/")
        lines.append(
            f"| {i} | {q}... | {r['retrieved_count']} | {r['elapsed_s']}s | {ans}... |"
        )

    lines += ["", "---", "", "## Full Answers", ""]

    for i, r in enumerate(results, 1):
        ents = r.get("entities", {})
        lines += [
            f"### Q{i}: {r['question']}",
            "",
            f"**Entities Extracted:**",
            f"- Tier: `{ents.get('tier')}` | County: `{ents.get('county')}` "
            f"| Company: `{ents.get('company')}` | OEM: `{ents.get('oem')}`",
            f"- Role: `{ents.get('role')}` | Keywords: `{ents.get('keywords')}`",
            f"- Aggregate: `{ents.get('aggregate')}` | Retrieved: `{r['retrieved_count']}` items | Time: `{r['elapsed_s']}s`",
            "",
            "**Answer:**",
            "",
            r["answer"],
            "",
            "---",
            "",
        ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Results saved to: {out_path}")
    return str(out_path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_evaluation()
    save_markdown(results)
    print(f"\nDone. Open scripts/evaluation_results.md to review all answers.")
