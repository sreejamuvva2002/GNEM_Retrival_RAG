"""Main runtime RAG pipeline orchestrator.

Flow:
  User question
  -> basic preprocessing
  -> query rewriting (structured entity extraction)
  -> vocabulary matching (resolve to row_ids)
  -> dense retrieval (pgvector over child chunks)
  -> sparse retrieval (BM25 over child chunks)
  -> hybrid fusion (Reciprocal Rank Fusion)
  -> vocabulary filter retrieval (exact parent fetch by row_ids)
  -> merge hybrid + vocabulary results
  -> build structured context with citation IDs
  -> call local LLM (qwen2.5:14b via Ollama)
  -> generate final grounded answer
  -> format citations
  -> log trace
  -> return RagResult
"""
from __future__ import annotations

import dataclasses
import time

from .schemas import CitationOutput, PipelineConfig, RagResult, RetrievalTrace
from .retrieval.retrieval_orchestrator import RetrievalOrchestrator, OrchestratorResult
from .generation.context_builder import build_context
from .generation.prompt_builder import build_prompt
from .generation.llm_client import generate_answer
from .generation.citation_formatter import format_citations
from .evaluation.trace_logger import build_trace


# Module-level singleton for the retrieval orchestrator (lazy init)
_orchestrator: RetrievalOrchestrator | None = None
_pipeline_config: PipelineConfig | None = None


def _get_orchestrator() -> RetrievalOrchestrator:
    """Lazily initialise and return the retrieval orchestrator."""
    global _orchestrator, _pipeline_config
    if _orchestrator is None:
        _pipeline_config = PipelineConfig()
        _orchestrator = RetrievalOrchestrator(pipeline_config=_pipeline_config)
    return _orchestrator


def _get_config() -> PipelineConfig:
    global _pipeline_config
    if _pipeline_config is None:
        _pipeline_config = PipelineConfig()
    return _pipeline_config


def run(question: str) -> RagResult:
    """Execute the full RAG pipeline from question to grounded answer.

    Args:
        question: The user's natural language question.

    Returns:
        RagResult with answer, citations, and full retrieval trace.
    """
    latency: dict[str, float] = {}
    errors: list[str] = []

    # Basic preprocessing
    question = question.strip()
    if not question:
        return RagResult(
            question=question,
            answer="Please provide a question.",
            trace=RetrievalTrace(question=question, errors=["Empty question"]),
        )

    # Stage 1-2: Orchestrated retrieval (hybrid + vocabulary)
    t0 = time.time()
    orchestrator = _get_orchestrator()
    try:
        orchestrator_result = orchestrator.search(question)
    except Exception as e:
        errors.append(f"Retrieval error: {e}")
        return RagResult(
            question=question,
            answer="An error occurred during retrieval. Please try again.",
            trace=RetrievalTrace(question=question, errors=errors),
        )
    latency["retrieval"] = time.time() - t0

    # Unpack for downstream compatibility
    fused_children = orchestrator_result.fused_children
    dense_results = orchestrator_result.dense_results
    bm25_results = orchestrator_result.bm25_results
    parent_contexts = orchestrator_result.parent_contexts

    if not parent_contexts:
        return RagResult(
            question=question,
            answer="No matching records found in the knowledge base for this question.",
            trace=RetrievalTrace(question=question, errors=["No parents fetched"]),
        )

    # Stage 3: Build context
    t0 = time.time()
    context_str, citation_map, included_parents = build_context(parent_contexts)
    latency["context_build"] = time.time() - t0

    # Stage 4: Build prompt and generate answer
    t0 = time.time()
    prompt = build_prompt(question, context_str)
    try:
        answer = generate_answer(prompt)
    except Exception as e:
        errors.append(f"LLM generation error: {e}")
        answer = "An error occurred during answer generation. Please check that the LLM is running."
    latency["llm_generation"] = time.time() - t0

    # Stage 5: Format citations
    t0 = time.time()
    citations = format_citations(answer, citation_map)
    latency["citation_format"] = time.time() - t0

    # Stage 6: Build trace
    trace = build_trace(
        question=question,
        dense_results=dense_results,
        bm25_results=bm25_results,
        fused_results=fused_children,
        fetched_parent_count=len(parent_contexts),
        included_parents=included_parents,
        context_sent=context_str,
        answer=answer,
        citations=citations,
        latency=latency,
        errors=errors,
    )

    # Add vocabulary filtering trace fields
    trace.structured_query = dataclasses.asdict(
        orchestrator_result.structured_query
    )
    trace.vocabulary_matches_count = orchestrator_result.vocabulary_matches.match_count
    trace.vocabulary_parents_count = len(orchestrator_result.vocabulary_parents)
    trace.vocabulary_used = orchestrator_result.vocabulary_used
    trace.rewrite_latency_ms = orchestrator_result.structured_query.rewrite_latency_ms

    return RagResult(
        question=question,
        answer=answer,
        citations=citations,
        parent_contexts_used=len(included_parents),
        retrieval_method="hybrid_rrf",
        trace=trace,
    )
