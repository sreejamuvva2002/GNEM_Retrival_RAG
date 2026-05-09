"""
Quick sanity check: runs 5 representative questions and prints results.
Usage: python -m georgia_ev_intelligence.scripts.smoke_test
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from georgia_ev_intelligence import pipeline

QUESTIONS = [
    "Show all Tier 1/2 suppliers in Georgia, list their EV Supply Chain Role and Product/Service",
    "Which Georgia companies are classified under Battery Cell or Battery Pack roles, and what tier is each assigned?",
    "Map all Thermal Management suppliers in Georgia and show which Primary OEMs they are linked to",
    "Which EV supply chain roles are served by only a single company in Georgia — single point of failure?",
    "Which county has the highest total employment across all companies in Georgia?",
]


def main():
    print("=" * 70)
    print("SMOKE TEST — 5 questions")
    print("=" * 70)

    passed = 0
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[Q{i}] {q}")
        result = pipeline.run(q)
        print(f"  Evidence rows : {result.evidence_count}")
        print(f"  Support level : {result.support_level}")
        print(f"  Intent        : {result.intent.get('type')}")
        print(f"  Risk          : {result.hallucination_risk}")
        print(f"  Answer        : {result.answer[:200]}{'...' if len(result.answer) > 200 else ''}")

        if result.evidence_count > 0:
            passed += 1
        else:
            print("  *** WARNING: no evidence rows retrieved ***")

    print(f"\n{'='*70}")
    print(f"RESULT: {passed}/{len(QUESTIONS)} questions had evidence rows")

    # Zero-hardcoding check: ensure no data values are hardcoded in source files.
    # We check for the pattern OUTSIDE this script (which legitimately uses them as test input).
    import subprocess, tempfile, os
    src_dir = Path(__file__).parent.parent
    py_files = subprocess.run(
        ["find", str(src_dir), "-name", "*.py", "-not", "-path", str(__file__)],
        capture_output=True, text=True
    ).stdout.strip().split("\n")
    domain_terms = ["Hyundai", "Troup County", "Kia Georgia", "Floyd County", "Battery America"]
    violations = []
    for fpath in py_files:
        if not fpath:
            continue
        try:
            lines = open(fpath).readlines()
            code_lines = [l for l in lines if not l.strip().startswith("#")]
            content = "".join(code_lines)
            for term in domain_terms:
                if term in content:
                    violations.append(f"{fpath}: '{term}'")
        except Exception:
            pass
    if violations:
        print("\n*** HARDCODING WARNING: found domain data values in source:")
        for v in violations:
            print(f"  {v}")
    else:
        print("\nZero-hardcoding check: PASSED (no domain data values found in source files)")


if __name__ == "__main__":
    main()
