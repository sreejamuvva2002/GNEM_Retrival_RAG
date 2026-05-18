"""
Run the RAG pipeline for a single question (with progress prints).

Usage:
  python -m georgia_ev_intelligence.runtime_pipeline.scripts.ask_question "Your question here"

Or from repo root with default test question:
  python -m georgia_ev_intelligence.runtime_pipeline.scripts.ask_question
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from georgia_ev_intelligence.shared.config import settings


def _save_query_analysis(question: str) -> Path:
    """Run query analysis and save the breakdown to an xlsx file.

    Saves to: outputs/single_question_analyze/<YYYYMMDD_HHMMSS>_analysis.xlsx
    Returns the path of the saved file.
    """
    from georgia_ev_intelligence.runtime_pipeline.query_analyzer.analyzer import QueryAnalyzer
    from georgia_ev_intelligence.runtime_pipeline.query_analyzer.vocabulary_repository import (
        PostgresVocabularyRepository,
    )

    repo = PostgresVocabularyRepository()
    analyzer = QueryAnalyzer(vocabulary_repository=repo)
    result = analyzer.analyze(question)

    # --- Build dataframes for each section ---

    # Sheet 1 – Perfect Keywords (vocabulary matches)
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

    # Sheet 3 – Common Words (stopwords / connectors)
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

    # --- Write xlsx ---
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
    question = (
        "Which companies are located in Fulton County?"
        if len(sys.argv) < 2
        else " ".join(sys.argv[1:])
    )

    print("Loading pipeline (first run may download/load embedding model — can take several minutes)...")
    sys.stdout.flush()
    t0 = time.time()

    from georgia_ev_intelligence.runtime_pipeline import pipeline

    print(f"Running pipeline ({time.time() - t0:.1f}s since start)...")
    sys.stdout.flush()

    result = pipeline.run(question)

    print(f"\nDone in {time.time() - t0:.1f}s total\n")
    print("Question:", result.question)
    print("-" * 60)
    print("Answer:\n")
    print(result.answer)
    print("-" * 60)
    print("Parents used:", result.parent_contexts_used)
    if result.trace.errors:
        print("Errors:", result.trace.errors)
    print("Retrieval counts:", {
        "dense": result.trace.dense_result_count,
        "bm25": result.trace.bm25_result_count,
        "hybrid": result.trace.hybrid_result_count,
    })

    print("\nAnalyzing query keywords...")
    sys.stdout.flush()
    xlsx_path = _save_query_analysis(question)
    print(f"Query analysis saved → {xlsx_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
