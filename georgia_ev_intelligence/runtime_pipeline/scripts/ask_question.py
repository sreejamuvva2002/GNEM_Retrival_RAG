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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
