"""Main runtime RAG pipeline orchestrator."""
from __future__ import annotations

import time

from .schemas import PipelineConfig, RagResult, RetrievalTrace
from .retrieval.retrieval_orchestrator import RetrievalOrchestrator
from .generation.context_builder import build_context
from .generation.prompt_builder import build_prompt
from .generation.llm_client import generate_answer
from .generation.citation_formatter import format_citations
from .evaluation.trace_logger import build_trace


_orchestrator: RetrievalOrchestrator | None = None


def _get_orchestrator() -> RetrievalOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RetrievalOrchestrator(pipeline_config=PipelineConfig())
    return _orchestrator


def run(question: str) -> RagResult:
    """Execute the full RAG pipeline from question to grounded answer."""
    latency: dict[str, float] = {}
    errors: list[str] = []

    question = question.strip()
    if not question:
        return RagResult(
            question=question,
            answer="Please provide a question.",
            trace=RetrievalTrace(question=question, errors=["Empty question"]),
        )

    t0 = time.time()
    try:
        retrieval = _get_orchestrator().search(question)
    except Exception as e:
        errors.append(f"Retrieval error: {e}")
        return RagResult(
            question=question,
            answer="An error occurred during retrieval. Please try again.",
            trace=RetrievalTrace(question=question, errors=errors),
        )
    latency["retrieval"] = time.time() - t0

    parent_contexts = retrieval.parent_contexts
    if not parent_contexts:
        return RagResult(
            question=question,
            answer="No matching records found in the knowledge base for this question.",
            trace=RetrievalTrace(
                question=question,
                errors=["No parents fetched"],
                dense_result_count=len(retrieval.dense_results),
                bm25_result_count=len(retrieval.bm25_results),
                hybrid_result_count=len(retrieval.fused_children),
            ),
        )

    t0 = time.time()
    context_str, citation_map, included_parents = build_context(parent_contexts)
    latency["context_build"] = time.time() - t0

    t0 = time.time()
    prompt = build_prompt(question, context_str)
    try:
        answer = generate_answer(prompt)
    except Exception as e:
        errors.append(f"LLM generation error: {e}")
        answer = "An error occurred during answer generation. Please check that the LLM is running."
    latency["llm_generation"] = time.time() - t0

    t0 = time.time()
    citations = format_citations(answer, citation_map)
    latency["citation_format"] = time.time() - t0

    trace = build_trace(
        question=question,
        dense_results=retrieval.dense_results,
        bm25_results=retrieval.bm25_results,
        fused_results=retrieval.fused_children,
        fetched_parent_count=len(parent_contexts),
        included_parents=included_parents,
        context_sent=context_str,
        answer=answer,
        citations=citations,
        latency=latency,
        errors=errors,
    )
    trace.dense_result_count = len(retrieval.dense_results)
    trace.bm25_result_count = len(retrieval.bm25_results)
    trace.hybrid_result_count = len(retrieval.fused_children)

    return RagResult(
        question=question,
        answer=answer,
        citations=citations,
        parent_contexts_used=len(included_parents),
        trace=trace,
    )
