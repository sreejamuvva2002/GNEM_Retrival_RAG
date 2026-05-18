"""Build the PreGenerationContextPackage by running retrieval on the original query."""
from __future__ import annotations

import logging
import time

from ..ClarificationOptionGenerator.models import ResolvedQueryContext
from ..retrieval.retrieval_orchestrator import RetrievalOrchestrator
from ..schemas import PipelineConfig
from .runtime_context import PreGenerationContextPackage

logger = logging.getLogger(__name__)


def build_pre_generation_context(
    query: str,
    resolved_context: ResolvedQueryContext,
    retrieval_orchestrator: RetrievalOrchestrator | None = None,
) -> PreGenerationContextPackage:
    """Run hybrid retrieval on the original query and build context package.

    Important: HybridRetriever.search() receives the raw original query.
    Clarification context is additional interpretation, NOT a rewritten query.
    No reranker exists in the codebase.
    """
    if retrieval_orchestrator is None:
        retrieval_orchestrator = RetrievalOrchestrator(pipeline_config=PipelineConfig())

    t0 = time.time()
    try:
        retrieval = retrieval_orchestrator.search(query)
    except Exception:
        logger.warning("Retrieval failed during pre-generation context build", exc_info=True)
        return PreGenerationContextPackage(
            original_query=query,
            resolved_query_context=resolved_context,
            retrieval_trace={"error": "retrieval_failed"},
            ready_for_generation=False,
        )
    retrieval_latency = time.time() - t0

    # No reranker exists — reranked_results stays None.
    parent_records = retrieval.parent_contexts

    trace = {
        "dense_result_count": len(retrieval.dense_results),
        "bm25_result_count": len(retrieval.bm25_results),
        "hybrid_result_count": len(retrieval.fused_children),
        "parent_context_count": len(parent_records),
        "reranker_used": False,
        "retrieval_latency_s": round(retrieval_latency, 3),
    }

    return PreGenerationContextPackage(
        original_query=query,
        resolved_query_context=resolved_context,
        retrieval_results=retrieval.fused_children,
        reranked_results=None,
        parent_records=parent_records,
        citations=[],
        retrieval_trace=trace,
        ready_for_generation=bool(parent_records),
    )
