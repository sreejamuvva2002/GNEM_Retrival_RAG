"""
Analyze a query, classify leftover phrases via LLM, interactively clarify
any ambiguous concepts, and save the results.

Usage:
  python -m georgia_ev_intelligence.runtime_pipeline.scripts.clarify_query "Your question here"

Flow:
  1. Analyzes the query and prints the breakdown (perfect / ambiguous / common words).
  2. Runs LLM phrase classifier on remaining unmatched phrases.
  3. If ambiguous_concept phrases exist, prompts the user for open-ended clarification.
  4. Saves the full analysis + clarifications to:
       outputs/clarifications_registered/<YYYYMMDD_HHMMSS>_clarification.xlsx
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from georgia_ev_intelligence.shared.config import settings


def _header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _print_analysis(analysis) -> None:
    """Print the query breakdown to the terminal."""
    print(f"\nQuery: {analysis.original_query}")
    print("=" * 60)

    _header("Perfect Keywords  (matched vocabulary terms)")
    if analysis.matched_vocabulary:
        for m in analysis.matched_vocabulary:
            tag = f"[{m.source_column}]"
            match_label = f"({m.match_type})" if m.match_type != "canonical" else ""
            canonical = f" -> '{m.canonical_value}'" if m.canonical_value != m.matched_text else ""
            print(f"  '{m.matched_text}'{canonical}  {tag} {match_label}".rstrip())
    else:
        print("  (none)")

    _header("Ambiguous Keywords  (meaningful but not in vocabulary)")
    if analysis.ambiguous_terms:
        for term in analysis.ambiguous_terms:
            print(f"  '{term}'")
    else:
        print("  (none)")

    _header("Common Words  (stopwords / connectors -- ignored)")
    if analysis.ignored_tokens:
        print("  " + ", ".join(f"'{t}'" for t in analysis.ignored_tokens))
    else:
        print("  (none)")

    _header("Detected Intent")
    print(f"  Operation     : {analysis.operation or '(none)'}")
    print(f"  Target entity : {analysis.target_entity or '(none)'}")


def _print_phrase_classification(phrase_result) -> None:
    """Print phrase classification results."""
    _header("LLM Phrase Classification")
    if not phrase_result.classified_phrases:
        print("  (no phrases to classify)")
        return
    for cp in phrase_result.classified_phrases:
        flag = " [NEEDS CLARIFICATION]" if cp.needs_clarification else ""
        print(f"  '{cp.phrase}' -> {cp.category.value}{flag}")
        if cp.reason:
            print(f"     Reason: {cp.reason}")
    if phrase_result.clarification_required:
        count = len(phrase_result.ambiguous_terms)
        print(f"\n  {count} phrase(s) need clarification.")
    else:
        print("\n  No clarification needed.")


def _print_resolved(context) -> None:
    """Print the resolved clarifications summary."""
    _header("Clarification Results")
    if not context.resolved_clarifications:
        print("  No clarifications recorded.")
        return
    for rc in context.resolved_clarifications:
        status = "IGNORED" if rc.ignored else f"-> {rc.meaning}"
        print(f"  '{rc.term}': {status}")
    if context.remaining_ambiguous_terms:
        print(f"\n  Still ambiguous: {', '.join(repr(t) for t in context.remaining_ambiguous_terms)}")


def _save_to_xlsx(analysis, phrase_result, context) -> Path:
    """Save the full analysis + clarifications to xlsx."""

    # Sheet 1 -- Summary
    df_summary = pd.DataFrame([
        {"Field": "Query", "Value": analysis.original_query},
        {"Field": "Normalized Query", "Value": analysis.normalized_query},
        {"Field": "Detected Operation", "Value": context.operation or "(none)"},
        {"Field": "Detected Target Entity", "Value": context.target_entity or "(none)"},
        {"Field": "Perfect Keywords Count", "Value": len(context.merged_matched_vocabulary)},
        {"Field": "Ambiguous Keywords Count", "Value": len(analysis.ambiguous_terms)},
        {"Field": "Common Words Count", "Value": len(analysis.ignored_tokens)},
        {"Field": "Clarifications Recorded", "Value": len(context.resolved_clarifications)},
        {"Field": "LLM Clarifications Count", "Value": sum(
            1 for cp in (phrase_result.classified_phrases or [])
            if cp.category.value != "ambiguous_concept"
        )},
        {"Field": "User Clarifications Count", "Value": len(context.resolved_clarifications)},
        {"Field": "Remaining Ambiguous Terms", "Value": len(context.remaining_ambiguous_terms)},
        {"Field": "Semantic Intent Terms", "Value": ", ".join(context.semantic_intent_terms) or "(none)"},
        {"Field": "Context Terms", "Value": ", ".join(context.context_terms) or "(none)"},
        {"Field": "Domain Signal Terms", "Value": ", ".join(context.domain_signal_terms) or "(none)"},
    ])

    # Sheet 2 -- Perfect Keywords (merged)
    vocab = context.merged_matched_vocabulary or analysis.matched_vocabulary
    if vocab:
        perfect_rows = [
            {
                "Matched Text": m.matched_text,
                "Canonical Value": m.canonical_value,
                "Source Column": m.source_column,
                "Match Type": m.match_type,
                "Confidence": round(m.confidence, 4),
            }
            for m in vocab
        ]
    else:
        perfect_rows = [{"Matched Text": "(none)", "Canonical Value": "", "Source Column": "", "Match Type": "", "Confidence": ""}]
    df_perfect = pd.DataFrame(perfect_rows)

    # Sheet 3 -- Phrase Classification
    if phrase_result.classified_phrases:
        class_rows = [
            {
                "Phrase": cp.phrase,
                "Category": cp.category.value,
                "Needs Clarification": cp.needs_clarification,
                "Question": cp.clarification_question or "",
                "Reason": cp.reason,
            }
            for cp in phrase_result.classified_phrases
        ]
    else:
        class_rows = [{"Phrase": "(none)", "Category": "", "Needs Clarification": "", "Question": "", "Reason": ""}]
    df_classification = pd.DataFrame(class_rows)

    # Sheet 4 -- All Clarifications (LLM + User)
    clarif_rows: list[dict[str, object]] = []

    # 4a. LLM clarifications: phrases the LLM resolved (non-ambiguous categories).
    if phrase_result.classified_phrases:
        for cp in phrase_result.classified_phrases:
            if cp.category.value == "ambiguous_concept":
                continue
            clarif_rows.append({
                "Term": cp.phrase,
                "Source": "LLM",
                "Category": cp.category.value,
                "Meaning": cp.reason,
                "Ignored": cp.category.value == "irrelevant_phrase",
            })

    # 4b. User clarifications: phrases the user resolved (ambiguous_concept).
    if context.resolved_clarifications:
        for rc in context.resolved_clarifications:
            clarif_rows.append({
                "Term": rc.term,
                "Source": "User",
                "Category": "ambiguous_concept",
                "Meaning": rc.meaning or "",
                "Ignored": rc.ignored,
            })

    if not clarif_rows:
        clarif_rows = [{"Term": "(none)", "Source": "", "Category": "", "Meaning": "", "Ignored": ""}]
    df_clarifications = pd.DataFrame(clarif_rows)

    # Sheet 5 -- Common Words
    df_common = pd.DataFrame(
        {"Common Word": analysis.ignored_tokens} if analysis.ignored_tokens else {"Common Word": ["(none)"]}
    )

    out_dir = settings.OUTPUTS_DIR / "clarifications_registered"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}_clarification.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_perfect.to_excel(writer, sheet_name="Perfect Keywords", index=False)
        df_classification.to_excel(writer, sheet_name="Phrase Classification", index=False)
        df_clarifications.to_excel(writer, sheet_name="All Clarifications", index=False)
        df_common.to_excel(writer, sheet_name="Common Words", index=False)

    return out_path


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m georgia_ev_intelligence.runtime_pipeline.scripts.clarify_query "<query>"')
        return 1

    query = " ".join(sys.argv[1:])

    print("Loading vocabulary (connecting to database)...")
    sys.stdout.flush()

    from georgia_ev_intelligence.runtime_pipeline.query_analyzer.analyzer import QueryAnalyzer
    from georgia_ev_intelligence.runtime_pipeline.query_analyzer.vocabulary_repository import (
        PostgresVocabularyRepository,
    )
    from georgia_ev_intelligence.runtime_pipeline.phrase_classifier import RemainingPhraseClassifier
    from georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator import (
        AnalysisMerger,
        ClarificationResolver,
        InMemoryClarificationStore,
        TerminalClarificationPrompter,
        TerminalClarificationWorkflow,
        ClarificationCancelledError,
    )

    # Wire up all components.
    repo = PostgresVocabularyRepository()
    analyzer = QueryAnalyzer(vocabulary_repository=repo)
    classifier = RemainingPhraseClassifier()
    store = InMemoryClarificationStore()
    prompter = TerminalClarificationPrompter()
    merger = AnalysisMerger()
    resolver = ClarificationResolver(
        store=store,
        query_analyzer=analyzer,
        phrase_classifier=classifier,
        merger=merger,
    )
    workflow = TerminalClarificationWorkflow(
        analyzer=analyzer,
        phrase_classifier=classifier,
        store=store,
        prompter=prompter,
        resolver=resolver,
    )

    # Step 1: print analysis breakdown (pre-warm the vocab index).
    analysis = analyzer.analyze(query)
    _print_analysis(analysis)

    # Step 2: run phrase classifier.
    print("\nClassifying remaining phrases via LLM...")
    sys.stdout.flush()
    phrase_result = classifier.classify(query, analysis)
    _print_phrase_classification(phrase_result)

    # Step 3: run clarification workflow.
    if not phrase_result.clarification_required:
        print("\nNo clarification needed.")
    else:
        count = len(phrase_result.ambiguous_terms)
        print(f"\n{count} phrase(s) need clarification. Starting clarification...\n")

    sys.stdout.flush()

    try:
        context = workflow.analyze_and_maybe_clarify(query)
    except ClarificationCancelledError:
        print("\nClarification cancelled. No file saved.")
        return 1

    # Step 4: print resolved summary.
    _print_resolved(context)

    # Step 5: save to xlsx.
    xlsx_path = _save_to_xlsx(analysis, phrase_result, context)
    print(f"\nClarification saved -> {xlsx_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
