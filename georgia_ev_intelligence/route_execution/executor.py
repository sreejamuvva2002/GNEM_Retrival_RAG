"""Route execution dispatcher (execution README §7).

Reads a validated ``FinalRoute`` (as a plain dict, exactly as stored in
``outputs/final_routes.jsonl``) and calls the matching executor. This module
never calls the router, never modifies the route, and never lets an LLM write
SQL — it only dispatches to the safe per-route executors.

When ``use_llm`` is set, the deterministic answer produced by an executor is
re-grounded through the local Ollama model using the already-retrieved
evidence. Direct routes (no_retrieval / clarification / out_of_domain) are
returned verbatim.
"""
from __future__ import annotations

import logging
from typing import Any

from . import answer_formatter as fmt
from .schemas import STATUS_SUCCESS, ExecutionResult
from .executors.direct import (
    execute_clarification,
    execute_no_retrieval,
    execute_out_of_domain,
)
from .executors.disruption import execute_disruption_analysis
from .executors.exact_lookup import execute_exact_lookup
from .executors.geo_search import execute_geo_search
from .executors.hybrid_search import execute_hybrid_search
from .executors.keyword_search import execute_keyword_search
from .executors.structured_sql import execute_structured_sql
from .executors.vector_search import execute_vector_search

logger = logging.getLogger(__name__)

# Routes whose answers may be re-grounded by the LLM (they carry real evidence).
_LLM_GROUNDABLE = {
    "structured_sql",
    "exact_lookup",
    "keyword_search",
    "vector_search",
    "hybrid_search",
    "geo_search",
    "disruption_analysis",
}

_DISPATCH = {
    "no_retrieval": execute_no_retrieval,
    "clarification_needed": execute_clarification,
    "out_of_domain": execute_out_of_domain,
    "structured_sql": execute_structured_sql,
    "exact_lookup": execute_exact_lookup,
    "keyword_search": execute_keyword_search,
    "vector_search": execute_vector_search,
    "hybrid_search": execute_hybrid_search,
    "geo_search": execute_geo_search,
    "disruption_analysis": execute_disruption_analysis,
}


def execute_route(final_route: dict[str, Any], *, use_llm: bool = False) -> ExecutionResult:
    """Execute one validated route dict and return an ``ExecutionResult``."""
    route = final_route.get("route")
    handler = _DISPATCH.get(route)
    if handler is None:
        return ExecutionResult.failure(
            route=str(route),
            reason=f"Unsupported route: {route!r}",
        )

    result = handler(final_route)

    if use_llm and result.status == STATUS_SUCCESS and route in _LLM_GROUNDABLE:
        question = final_route.get("question") or final_route.get("query_focus") or ""
        result.answer = fmt.llm_answer(question, result.evidence, result.answer)

    return result
