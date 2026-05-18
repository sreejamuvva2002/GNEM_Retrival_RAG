"""
Show a breakdown of how a user query is analyzed — without running the full pipeline.
Also saves the breakdown to outputs/single_question_analyze/<timestamp>_analysis.xlsx.

Usage:
  python -m georgia_ev_intelligence.runtime_pipeline.scripts.analyze_query "Your query here"

Outputs:
  - Perfect keywords  : vocabulary-matched terms (exact / alias / normalized)
  - Ambiguous keywords: meaningful tokens not found in the vocabulary
  - Common words      : stopwords / connectors that are ignored
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


def _save_to_xlsx(result) -> Path:
    """Save query analysis breakdown to outputs/single_question_analyze/."""
    # Sheet 1 – Perfect Keywords
    if result.matched_vocabulary:
        perfect_rows = [
            {
                "Matched Text": m.matched_text,
                "Canonical Value": m.canonical_value,
                "Source Column": m.source_column,
                "Match Type": m.match_type,
                "Confidence": round(m.confidence, 4),
            }
            for m in result.matched_vocabulary
        ]
    else:
        perfect_rows = [{"Matched Text": "(none)", "Canonical Value": "", "Source Column": "", "Match Type": "", "Confidence": ""}]
    df_perfect = pd.DataFrame(perfect_rows)

    # Sheet 2 – Ambiguous Keywords
    df_ambiguous = pd.DataFrame(
        {"Ambiguous Term": result.ambiguous_terms} if result.ambiguous_terms else {"Ambiguous Term": ["(none)"]}
    )

    # Sheet 3 – Common Words
    df_common = pd.DataFrame(
        {"Common Word": result.ignored_tokens} if result.ignored_tokens else {"Common Word": ["(none)"]}
    )

    # Sheet 4 – Summary
    df_summary = pd.DataFrame([
        {"Field": "Query", "Value": result.original_query},
        {"Field": "Normalized Query", "Value": result.normalized_query},
        {"Field": "Detected Operation", "Value": result.operation or "(none)"},
        {"Field": "Detected Target Entity", "Value": result.target_entity or "(none)"},
        {"Field": "Perfect Keywords Count", "Value": len(result.matched_vocabulary)},
        {"Field": "Ambiguous Keywords Count", "Value": len(result.ambiguous_terms)},
        {"Field": "Common Words Count", "Value": len(result.ignored_tokens)},
    ])

    out_dir = settings.OUTPUTS_DIR / "single_question_analyze"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}_analysis.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_perfect.to_excel(writer, sheet_name="Perfect Keywords", index=False)
        df_ambiguous.to_excel(writer, sheet_name="Ambiguous Keywords", index=False)
        df_common.to_excel(writer, sheet_name="Common Words", index=False)

    return out_path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m georgia_ev_intelligence.runtime_pipeline.scripts.analyze_query \"<query>\"")
        return 1

    query = " ".join(sys.argv[1:])

    print("Loading vocabulary (connecting to database)...")
    sys.stdout.flush()

    from georgia_ev_intelligence.runtime_pipeline.query_analyzer.analyzer import QueryAnalyzer
    from georgia_ev_intelligence.runtime_pipeline.query_analyzer.vocabulary_repository import (
        PostgresVocabularyRepository,
    )

    repo = PostgresVocabularyRepository()
    analyzer = QueryAnalyzer(vocabulary_repository=repo)

    result = analyzer.analyze(query)

    print(f"\nQuery: {result.original_query}")
    print("=" * 60)

    # --- Perfect keywords ---
    _header("Perfect Keywords  (matched vocabulary terms)")
    if result.matched_vocabulary:
        for match in result.matched_vocabulary:
            tag = f"[{match.source_column}]"
            match_label = f"({match.match_type})" if match.match_type != "canonical" else ""
            canonical = (
                f" → '{match.canonical_value}'" if match.canonical_value != match.matched_text else ""
            )
            print(f"  '{match.matched_text}'{canonical}  {tag} {match_label}".rstrip())
    else:
        print("  (none)")

    # --- Ambiguous keywords ---
    _header("Ambiguous Keywords  (meaningful but not in vocabulary)")
    if result.ambiguous_terms:
        for term in result.ambiguous_terms:
            print(f"  '{term}'")
    else:
        print("  (none)")

    # --- Common words ---
    _header("Common Words  (stopwords / connectors — ignored)")
    if result.ignored_tokens:
        print("  " + ", ".join(f"'{t}'" for t in result.ignored_tokens))
    else:
        print("  (none)")

    # --- Intent summary ---
    _header("Detected Intent")
    print(f"  Operation     : {result.operation or '(none)'}")
    print(f"  Target entity : {result.target_entity or '(none)'}")

    # --- Phrase classification (if ambiguous terms exist) ---
    if result.ambiguous_terms:
        _header("LLM Phrase Classification")
        try:
            from georgia_ev_intelligence.runtime_pipeline.phrase_classifier import (
                RemainingPhraseClassifier,
            )
            classifier = RemainingPhraseClassifier()
            phrase_result = classifier.classify(query, result)
            for cp in phrase_result.classified_phrases:
                flag = " [NEEDS CLARIFICATION]" if cp.needs_clarification else ""
                print(f"  '{cp.phrase}' -> {cp.category.value}{flag}")
            if phrase_result.clarification_required:
                print(f"\n  {len(phrase_result.ambiguous_terms)} phrase(s) need clarification.")
            else:
                print("\n  No clarification needed.")
        except Exception as exc:
            print(f"  (phrase classification unavailable: {exc})")

    # --- Save to xlsx ---
    xlsx_path = _save_to_xlsx(result)
    print(f"\nAnalysis saved -> {xlsx_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
