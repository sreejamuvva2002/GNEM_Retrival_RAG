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
        "Perfect Keywords (Resolver)",
        "Candidate Keywords",
        "Ambiguous Words Identified",
        "DeepSeek Guesses on Ambiguous Words",
        "Identified Argument Operations",
        "Deterministic Filters",
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

        # Keyword resolution debug output
        kw_res = getattr(result, "keyword_resolution", {})
        perfect_kws = kw_res.get("perfect", [])
        candidate_kws = kw_res.get("candidate", [])
        rejected_kws = kw_res.get("rejected", [])
        det_filters = kw_res.get("deterministic_filters", {})
        print(f"  KW Perfect    : {[k['value'] + ' [' + k['col'] + ']' for k in perfect_kws] or '(none)'}")
        print(f"  KW Candidate  : {[k['value'] + ' [' + k['col'] + ']' for k in candidate_kws[:5]] or '(none)'}")
        print(f"  KW Rejected   : {[k['value'] + ' (' + k['reason'] + ')' for k in rejected_kws[:3]] or '(none)'}")
        print(f"  Det Filters   : {det_filters or '(none)'}")
        print(f"  All Filters   : {result.filters_applied or '(none)'}")

        if result.evidence_count > 0:
            passed += 1
        else:
            print("  *** WARNING: no evidence rows retrieved ***")

        human_answer = _best_human_answer(q, qa_pairs)
        f1 = round(token_f1(result.answer, human_answer), 4) if human_answer != "(not in QA set)" else 0.0

        # Format perfect keywords for Excel
        perfect_kw_str = "; ".join(
            f"{k['value']} [{k['col']}] ({k['type']})"
            for k in perfect_kws
        ) or "(none)"
        candidate_kw_str = "; ".join(
            f"{k['value']} [{k['col']}]"
            for k in candidate_kws[:5]
        ) or "(none)"

        excel_rows.append({
            "#": i,
            "Question": result.question,
            "Perfect Keywords (Resolver)": perfect_kw_str,
            "Candidate Keywords": candidate_kw_str,
            "Ambiguous Words Identified": ", ".join(result.unmatched_words) or "(none)",
            "DeepSeek Guesses on Ambiguous Words": _format_mapped_phrases(result),
            "Identified Argument Operations": _format_intent(result.intent),
            "Deterministic Filters": "; ".join(f"{c}: {', '.join(v)}" for c, v in det_filters.items()) or "(none)",
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


# ── Structural assertion tests ────────────────────────────────────────────────

STRUCTURAL_TESTS = [
    {
        "question": "Show all Tier 1/2 suppliers in Georgia",
        "checks": {
            "requires_exhaustive": True,
            "no_tier_in_role_column": True,
            "georgia_is_scope": True,
        },
    },
    {
        "question": "Which roles are single point of failure?",
        "checks": {
            "operation_type": "spof",
            "no_county_keyword": True,
            "no_location_filter": True,
        },
    },
    {
        "question": "Show Battery Cell or Battery Pack companies",
        "checks": {
            "both_terms_in_filters_or_queries": ["Battery Cell", "Battery Pack"],
        },
    },
    {
        "question": "Thermal Management suppliers",
        "checks": {
            "term_in_valid_column": {"term": "Thermal Management", "valid_columns": ["ev_supply_chain_role", "product_service"]},
        },
    },
    {
        "question": "Which county has the highest total employment?",
        "checks": {
            "operation_type": "aggregate_sum",
        },
    },
]


def _norm_col(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")


def _run_structural_tests():
    """
    Run structural assertion tests.

    These tests validate correct pipeline behavior without hardcoding
    specific company names, row counts, or answer text.
    """
    from georgia_ev_intelligence.operation_detector import detect_operation
    from georgia_ev_intelligence.term_matcher import _is_tier_compatible_column

    print(f"\n{'='*70}")
    print("STRUCTURAL ASSERTION TESTS")
    print("="*70)

    passed = 0
    failed = 0

    for test in STRUCTURAL_TESTS:
        q = test["question"]
        checks = test["checks"]
        print(f"\n  TEST: {q}")

        result = pipeline.run(q)
        det_op = detect_operation(q)
        failures: list[str] = []

        # Check: requires_exhaustive
        if "requires_exhaustive" in checks:
            debug = getattr(result, "debug_info", {})
            is_exhaustive = debug.get("requires_exhaustive_retrieval", False)
            if is_exhaustive != checks["requires_exhaustive"]:
                failures.append(
                    f"requires_exhaustive: expected={checks['requires_exhaustive']}, got={is_exhaustive}"
                )

        # Check: no tier-derived filter in non-tier-compatible column
        if checks.get("no_tier_in_role_column"):
            for col, vals in result.filters_applied.items():
                if not _is_tier_compatible_column(col):
                    for v in vals:
                        if re.search(r"\btier\s*\d", v.lower()):
                            failures.append(
                                f"tier filter in incompatible column: {col}={v}"
                            )

        # Check: Georgia is dataset scope, not exact location filter
        if checks.get("georgia_is_scope"):
            for col, vals in result.filters_applied.items():
                norm = _norm_col(col)
                if any(x in norm for x in ("location", "county", "city")):
                    for v in vals:
                        if v.lower().strip() in ("georgia", "ga"):
                            failures.append(
                                f"Georgia used as exact location filter: {col}={v}"
                            )

        # Check: operation type
        if "operation_type" in checks:
            expected_op = checks["operation_type"]
            actual_op = det_op.get("type", "none")
            if actual_op != expected_op:
                failures.append(
                    f"operation_type: expected={expected_op}, got={actual_op}"
                )

        # Check: no county keyword for SPOF
        if checks.get("no_county_keyword"):
            for col in result.filters_applied:
                if "county" in col.lower():
                    failures.append(f"unexpected county filter: {col}")

        # Check: no location filter for SPOF
        if checks.get("no_location_filter"):
            for col in result.filters_applied:
                norm = _norm_col(col)
                if any(x in norm for x in ("location", "county", "city", "state")):
                    failures.append(f"unexpected location filter: {col}")

        # Check: both terms present in filters or rewritten queries
        if "both_terms_in_filters_or_queries" in checks:
            terms = checks["both_terms_in_filters_or_queries"]
            all_filter_vals = " ".join(
                v for vals in result.filters_applied.values() for v in vals
            ).lower()
            all_queries = " ".join(
                stage2_q for stage2_q in (
                    [result.rewritten_question, result.question]
                    if result.rewritten_question else [result.question]
                )
            ).lower()
            search_text = all_filter_vals + " " + all_queries
            for term in terms:
                if term.lower() not in search_text:
                    failures.append(f"missing term in filters/queries: {term}")

        # Check: term in valid column
        if "term_in_valid_column" in checks:
            spec = checks["term_in_valid_column"]
            term = spec["term"].lower()
            valid_cols = spec["valid_columns"]
            found_in_valid = False
            for col, vals in result.filters_applied.items():
                if _norm_col(col) in valid_cols:
                    if any(term in v.lower() for v in vals):
                        found_in_valid = True
            # Also check evidence rows
            if not found_in_valid and result.evidence_count > 0:
                for row in result.evidence_rows[:5]:
                    for vc in valid_cols:
                        for col_name, col_val in row.items():
                            if _norm_col(col_name) == vc and term in str(col_val).lower():
                                found_in_valid = True
                                break
            if not found_in_valid:
                failures.append(
                    f"term '{spec['term']}' not found in valid columns: {valid_cols}"
                )

        if failures:
            failed += 1
            print(f"    \u274c FAILED:")
            for f in failures:
                print(f"       - {f}")
        else:
            passed += 1
            print(f"    \u2705 PASSED")

        # Print debug info
        print(f"    Operation: {det_op.get('type', 'none')}")
        print(f"    Filters: {result.filters_applied}")
        print(f"    Intent: {result.intent.get('type', 'unknown')}")

    print(f"\n{'='*70}")
    print(f"STRUCTURAL TESTS: {passed} passed, {failed} failed out of {len(STRUCTURAL_TESTS)}")
    return failed == 0


if __name__ == "__main__":
    main()
    _run_structural_tests()