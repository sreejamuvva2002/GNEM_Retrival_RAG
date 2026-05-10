"""
Quick sanity check: runs 5 representative questions and prints results.
Also saves a timestamped Excel file to outputs/smoke_test/ after each run.
Usage: python -m georgia_ev_intelligence.scripts.smoke_test
"""
import re
import sys
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from georgia_ev_intelligence import pipeline, config
from georgia_ev_intelligence.evaluator import load_qa, token_f1

QUESTIONS = [
    "Show all Tier 1/2 suppliers in Georgia, list their EV Supply Chain Role and Product/Service",
    "Which Georgia companies are classified under Battery Cell or Battery Pack roles, and what tier is each assigned?",
    "Map all Thermal Management suppliers in Georgia and show which Primary OEMs they are linked to",
    "Which EV supply chain roles are served by only a single company in Georgia — single point of failure?",
    "Which county has the highest total employment across all companies in Georgia?",
]


def _best_human_answer(question: str, qa_pairs: list[dict]) -> str:
    q_tokens = {w for w in re.split(r"\W+", question.lower()) if len(w) >= 3}
    best_score, best_answer = 0, "(not in QA set)"
    for item in qa_pairs:
        a_tokens = {w for w in re.split(r"\W+", item["question"].lower()) if len(w) >= 3}
        score = len(q_tokens & a_tokens)
        if score > best_score:
            best_score, best_answer = score, item["human_answer"]
    return best_answer if best_score >= 5 else "(not in QA set)"


def _format_filters(result) -> str:
    if result.filters_applied:
        return "; ".join(
            f"{col}: {', '.join(vals)}" for col, vals in result.filters_applied.items()
        )
    if result.stage2_explicit_filters:
        return "(LLM) " + "; ".join(
            f"{col}: {val}" for col, val in result.stage2_explicit_filters.items()
        )
    if result.stage2_target_columns:
        return "(LLM target cols) " + ", ".join(result.stage2_target_columns)
    return "(none)"


def _format_intent(intent: dict) -> str:
    itype = intent.get("type", "unknown")
    direction = intent.get("direction", "")
    n = intent.get("n", "")
    parts = [itype]
    if direction:
        parts.append(direction)
    if n:
        parts.append(f"top {n}")
    return " / ".join(parts)


def _format_mapped_phrases(result) -> str:
    phrases = result.stage2_mapped_phrases
    if not phrases:
        return "(not available)"
    parts = []
    for p in phrases:
        user_phrase = p.get("user_phrase", "")
        kb_terms = p.get("kb_supported_terms", [])
        confidence = p.get("confidence", "")
        terms_str = ", ".join(kb_terms) if kb_terms else "—"
        parts.append(f"{user_phrase} → [{terms_str}] ({confidence})")
    return "; ".join(parts)


def _save_excel(rows: list[dict]) -> Path:
    config.SMOKE_TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.SMOKE_TEST_OUTPUTS_DIR / f"smoke_test_{timestamp}.xlsx"
    df = pd.DataFrame(rows, columns=[
        "#",
        "Question",
        "Ambiguous Words Identified",
        "DeepSeek Guesses on Ambiguous Words",
        "Identified Argument Operations",
        "Perfect Keywords Identified",
        "Retrieved Filters",
        "Modified Question",
        "Final Answer",
        "Answer Correctness (F1)",
        "Human Validated Answer",
    ])
    df.to_excel(path, index=False)
    return path


def main():
    print("=" * 70)
    print("SMOKE TEST — 5 questions")
    print("=" * 70)

    # Load human QA pairs for fuzzy matching
    qa_pairs: list[dict] = []
    try:
        qa_pairs = load_qa()
    except Exception:
        pass  # If QA file unavailable, F1 stays 0 and human answer stays "(not in QA set)"

    passed = 0
    excel_rows: list[dict] = []

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

        human_answer = _best_human_answer(q, qa_pairs)
        f1 = round(token_f1(result.answer, human_answer), 4) if human_answer != "(not in QA set)" else 0.0

        excel_rows.append({
            "#": i,
            "Question": result.question,
            "Ambiguous Words Identified": ", ".join(result.unmatched_words) or "(none)",
            "DeepSeek Guesses on Ambiguous Words": _format_mapped_phrases(result),
            "Identified Argument Operations": _format_intent(result.intent),
            "Perfect Keywords Identified": ", ".join(result.key_terms_matched) or "(none)",
            "Retrieved Filters": _format_filters(result),
            "Modified Question": result.rewritten_question or "(no rewrite)",
            "Final Answer": result.answer,
            "Answer Correctness (F1)": f1,
            "Human Validated Answer": human_answer,
        })

    print(f"\n{'='*70}")
    print(f"RESULT: {passed}/{len(QUESTIONS)} questions had evidence rows")

    # Save Excel
    excel_path = _save_excel(excel_rows)
    print(f"Excel saved  : {excel_path}")

    # Zero-hardcoding check
    import subprocess
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