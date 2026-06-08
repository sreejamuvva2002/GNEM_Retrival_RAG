"""Non-retrieval routes: no_retrieval, clarification_needed, out_of_domain.

These never touch the database. They translate the router's decision directly
into an ``ExecutionResult`` so the driver can record a uniform output row.
"""
from __future__ import annotations

from typing import Any

from ..schemas import STATUS_SUCCESS, ExecutionResult

_OUT_OF_DOMAIN_MESSAGE = "This question is outside the scope of the knowledge base."


def execute_no_retrieval(final_route: dict[str, Any]) -> ExecutionResult:
    reason = final_route.get("reason") or "No retrieval was required for this question."
    return ExecutionResult(
        route="no_retrieval",
        status=STATUS_SUCCESS,
        answer=reason,
        evidence={"type": "direct", "content": reason},
    )


def execute_clarification(final_route: dict[str, Any]) -> ExecutionResult:
    clarification = final_route.get("clarification") or {}
    message = ""
    if isinstance(clarification, dict):
        message = clarification.get("message") or clarification.get("question") or ""
    answer = message or "Could you clarify your question? More detail is needed to answer it."
    return ExecutionResult(
        route="clarification_needed",
        status=STATUS_SUCCESS,
        answer=answer,
        evidence={"type": "clarification", "clarification": clarification},
    )


def execute_out_of_domain(final_route: dict[str, Any]) -> ExecutionResult:
    return ExecutionResult(
        route="out_of_domain",
        status=STATUS_SUCCESS,
        answer=_OUT_OF_DOMAIN_MESSAGE,
        evidence={"type": "out_of_domain", "content": _OUT_OF_DOMAIN_MESSAGE},
    )
