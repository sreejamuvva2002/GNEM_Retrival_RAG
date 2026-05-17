"""
Runtime pipeline smoke test.

Verifies each stage of the hybrid retrieval pipeline:
  1. Database connection
  2. Dense pgvector retrieval
  3. BM25 retrieval
  4. Hybrid RRF fusion
  5. Parent chunk fetching
  6. Context building
  7. LLM answer generation
  8. Citation output

Usage: python -m georgia_ev_intelligence.runtime_pipeline.scripts.smoke_test
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import psycopg2

from georgia_ev_intelligence.shared import config


TEST_QUESTION = "Which companies are located in Fulton County?"


def main():
    print("=" * 60)
    print("RUNTIME PIPELINE SMOKE TEST")
    print("=" * 60)
    print(f"  Question: {TEST_QUESTION}")
    print(f"  NEON_DATABASE_URL is set: {'yes' if config.NEON_DATABASE_URL else 'no'}")
    print()

    results: dict[str, bool] = {}

    # Shared state passed between stages
    dense_results = None
    bm25_results = None
    fused_results = None
    parent_contexts = None
    context_str = None
    citation_map: dict = {}
    answer = None

    # 1. Database connection
    passed, msg = _run_stage("DB Connection", lambda: _test_db_connection())
    results["db_connection"] = passed

    # 2. Dense retrieval
    def _dense():
        nonlocal dense_results
        from georgia_ev_intelligence.runtime_pipeline.retrieval.dense_pgvector_retriever import (
            DensePgvectorRetriever,
        )
        dense = DensePgvectorRetriever()
        dense_results = dense.search(TEST_QUESTION, top_k=10)
        assert len(dense_results) > 0, "No dense results returned"
        return f"{len(dense_results)} results, top={dense_results[0].score:.4f}"

    passed, msg = _run_stage("Dense Retrieval", _dense)
    results["dense_retrieval"] = passed

    # 3. BM25 retrieval
    def _bm25():
        nonlocal bm25_results
        from georgia_ev_intelligence.runtime_pipeline.retrieval.bm25_retriever import BM25Retriever
        bm25 = BM25Retriever()
        bm25_results = bm25.search(TEST_QUESTION, top_k=10)
        assert len(bm25_results) > 0, "No BM25 results returned"
        return f"{len(bm25_results)} results, top={bm25_results[0].score:.4f}"

    passed, msg = _run_stage("BM25 Retrieval", _bm25)
    results["bm25_retrieval"] = passed

    # 4. Hybrid fusion
    def _hybrid():
        nonlocal fused_results, dense_results, bm25_results
        from georgia_ev_intelligence.runtime_pipeline.retrieval.hybrid_retriever import HybridRetriever
        hybrid = HybridRetriever()
        fused_results, dense_results, bm25_results = hybrid.search(TEST_QUESTION)
        assert len(fused_results) > 0, "No fused results returned"
        return f"{len(fused_results)} fused, top={fused_results[0].rrf_score:.6f}"

    passed, msg = _run_stage("Hybrid RRF Fusion", _hybrid)
    results["hybrid_fusion"] = passed

    # 5. Parent fetching
    def _parents():
        nonlocal parent_contexts
        from georgia_ev_intelligence.runtime_pipeline.retrieval.parent_fetcher import fetch_parents
        assert fused_results and dense_results and bm25_results, "Prior retrieval stages failed"
        parent_contexts = fetch_parents(fused_results, dense_results, bm25_results)
        assert len(parent_contexts) > 0, "No parents fetched"
        top = parent_contexts[0]
        return f"{len(parent_contexts)} parents, top={top.metadata.get('company', '?')}"

    passed, msg = _run_stage("Parent Fetching", _parents)
    results["parent_fetch"] = passed

    # 6. Context building
    included_parents = None

    def _context():
        nonlocal context_str, citation_map, included_parents
        from georgia_ev_intelligence.runtime_pipeline.generation.context_builder import build_context
        assert parent_contexts, "Parent fetching stage failed"
        context_str, citation_map, included_parents = build_context(parent_contexts)
        assert "[S1]" in context_str, "Citation ID [S1] not found in context"
        assert len(included_parents) == len(citation_map), (
            f"included_parents ({len(included_parents)}) != citation_map ({len(citation_map)})"
        )
        assert len(included_parents) <= len(parent_contexts), (
            f"included ({len(included_parents)}) > fetched ({len(parent_contexts)})"
        )
        return f"{len(included_parents)} included (of {len(parent_contexts)} fetched), {len(context_str)} chars"

    passed, msg = _run_stage("Context Building", _context)
    results["context_build"] = passed

    # 7. LLM generation
    def _llm():
        nonlocal answer
        from georgia_ev_intelligence.runtime_pipeline.generation.prompt_builder import build_prompt
        from georgia_ev_intelligence.runtime_pipeline.generation.llm_client import generate_answer
        assert context_str, "Context building stage failed"
        prompt = build_prompt(TEST_QUESTION, context_str)
        answer = generate_answer(prompt)
        assert len(answer) > 10, f"Answer too short: {answer!r}"
        return f"{len(answer)} chars"

    passed, msg = _run_stage("LLM Generation", _llm)
    results["llm_generation"] = passed

    # 8. Citation + trace consistency
    def _citations():
        from georgia_ev_intelligence.runtime_pipeline.generation.citation_formatter import (
            format_citations,
        )
        from georgia_ev_intelligence.runtime_pipeline.evaluation.trace_logger import build_trace
        from georgia_ev_intelligence.runtime_pipeline.evaluation.ragas_runner import prepare_sample
        from georgia_ev_intelligence.runtime_pipeline.schemas import RagResult, CitationOutput

        assert citation_map, "Context building stage failed"
        test_answer = answer or "Company in [S1] and [S2]."
        citations = format_citations(test_answer, citation_map)

        # all_source_records must equal included parents (sent to LLM)
        assert len(citations.all_source_records) == len(included_parents or []), (
            f"all_source_records ({len(citations.all_source_records)}) != "
            f"included_parents ({len(included_parents or [])})"
        )

        # Build trace and verify parent_chunk_texts == included parents
        trace = build_trace(
            question=TEST_QUESTION,
            dense_results=dense_results or [],
            bm25_results=bm25_results or [],
            fused_results=fused_results or [],
            fetched_parent_count=len(parent_contexts or []),
            included_parents=included_parents or [],
            context_sent=context_str or "",
            answer=test_answer,
            citations=citations,
            latency={},
        )
        assert len(trace.parent_chunk_texts) == len(included_parents or []), (
            f"trace.parent_chunk_texts ({len(trace.parent_chunk_texts)}) != "
            f"included_parents ({len(included_parents or [])})"
        )
        assert len(trace.parent_chunk_texts) <= trace.fetched_parent_count, (
            f"parent_chunk_texts ({len(trace.parent_chunk_texts)}) > "
            f"fetched_parent_count ({trace.fetched_parent_count})"
        )

        # Verify RAGAS contexts match trace.parent_chunk_texts
        mock_result = RagResult(
            question=TEST_QUESTION,
            answer=test_answer,
            citations=citations,
            parent_contexts_used=len(included_parents or []),
            trace=trace,
        )
        ragas_sample = prepare_sample(mock_result)
        assert len(ragas_sample["contexts"]) == len(trace.parent_chunk_texts), (
            f"RAGAS contexts ({len(ragas_sample['contexts'])}) != "
            f"trace.parent_chunk_texts ({len(trace.parent_chunk_texts)})"
        )

        return (
            f"used={len(citations.used_citations)}, "
            f"all={len(citations.all_source_records)}, "
            f"trace_texts={len(trace.parent_chunk_texts)}, "
            f"ragas_ctx={len(ragas_sample['contexts'])}"
        )

    passed, msg = _run_stage("Citation + Trace Consistency", _citations)
    results["citation_format"] = passed

    # Summary
    print()
    print("=" * 60)
    passed_count = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"RESULTS: {passed_count}/{total} stages passed")

    if passed_count == total:
        print("All stages PASSED. Pipeline is operational.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"FAILED stages: {failed}")

    print("=" * 60)

    # Print answer preview
    if answer:
        print(f"\nAnswer preview (first 500 chars):")
        print("-" * 40)
        print(answer[:500])
        print("-" * 40)

    return passed_count == total


def _test_db_connection() -> str:
    conn = psycopg2.connect(config.NEON_DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM parent_chunks")
    parent_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM child_chunks")
    child_count = cur.fetchone()[0]
    conn.close()
    assert parent_count > 0, "No parent chunks found"
    assert child_count > 0, "No child chunks found"
    return f"{parent_count} parents, {child_count} children"


def _run_stage(label: str, fn) -> tuple[bool, str]:
    """Run a test stage, print result, and return (passed, detail_msg)."""
    print(f"  [{label}] ... ", end="", flush=True)
    t0 = time.time()
    try:
        detail = fn()
        elapsed = time.time() - t0
        print(f"PASS ({elapsed:.2f}s) ({detail})")
        return True, detail
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAIL ({elapsed:.2f}s)")
        print(f"       Error: {e}")
        return False, str(e)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
