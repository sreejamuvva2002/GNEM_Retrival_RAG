"""Route-aware chat service: answer a question through the routing pipeline.

Unlike :class:`ChatService` (which calls the hybrid-retrieval orchestrator
directly), this service drives the *validated route* pipeline end-to-end:

    question
      -> build_default_route_service().route(question)   # router + validator
      -> execute_route(final_route, use_llm=True)        # safe per-route executor
      -> grounded answer

so the UI uses exactly the same contract that produces
``outputs/final_routes.jsonl`` / ``outputs/execution_results.jsonl``. It satisfies
the same ``IChatService`` protocol as :class:`ChatService`, so it drops straight
into the existing dispatcher / loading-card flow.

The executor returns an ``evidence`` dict (not ``ParentContext`` objects), so
``parent_contexts`` is left empty and the route/evidence summary is surfaced via
``trace`` instead. The map pane continues to work off the raw query.
"""
from __future__ import annotations

from typing import Any, Callable

from georgia_ev_intelligence.route_execution import execute_route
from georgia_ev_intelligence.route_execution.schemas import STATUS_FAILED

from .interfaces import ChatResult, IChatService


class RouteChatService(IChatService):
    """Concrete chat service backed by the router/validator + route executor."""

    def __init__(
        self,
        route_service_factory: Callable,
        execute_route_fn: Callable[..., Any] = execute_route,
        use_llm: bool = True,
    ) -> None:
        self._route_service_factory = route_service_factory
        self._execute_route = execute_route_fn
        self._use_llm = use_llm
        self._route_service = None

    def _route_service_lazy(self):
        if self._route_service is None:
            self._route_service = self._route_service_factory()
        return self._route_service

    def answer(self, query: str, on_step: Callable[[str], None] | None = None) -> ChatResult:
        def _step(name: str) -> None:
            if on_step is not None:
                try:
                    on_step(name)
                except Exception:
                    pass

        query = (query or "").strip()
        if not query:
            return ChatResult(answer="", parent_contexts=[], trace={}, error="Empty question.")

        # 1. Route: normalize -> pre/LLM router -> validator -> FinalRoute.
        try:
            _step("retrieval")
            final_route = self._route_service_lazy().route(query)
            route_dict = final_route.model_dump(mode="json")
            _step("rerank")
        except Exception as exc:
            return ChatResult(
                answer="", parent_contexts=[], trace={}, error=f"Routing failed: {exc}"
            )

        # 2. Execute the validated route (and re-ground the answer via Ollama).
        try:
            _step("generation")
            result = self._execute_route(route_dict, use_llm=self._use_llm)
        except Exception as exc:
            return ChatResult(
                answer="",
                parent_contexts=[],
                trace=_route_trace(route_dict, None),
                error=f"Execution failed: {exc}",
            )

        trace = _route_trace(route_dict, result)
        if result.status == STATUS_FAILED:
            return ChatResult(
                answer=result.answer or "",
                parent_contexts=[],
                trace=trace,
                error=result.error or "Route execution failed.",
            )

        return ChatResult(
            answer=result.answer or "(no answer returned)",
            parent_contexts=[],
            trace=trace,
        )


def _route_trace(route_dict: dict[str, Any], result) -> dict[str, Any]:
    """A compact, JSON-serialisable summary of how the answer was produced."""
    evidence = getattr(result, "evidence", {}) or {}
    payload = evidence.get(evidence.get("type", ""), None) if isinstance(evidence, dict) else None
    evidence_count = len(payload) if isinstance(payload, (list, tuple)) else None
    return {
        "route": route_dict.get("route"),
        "operation": route_dict.get("operation"),
        "route_source": route_dict.get("route_source"),
        "resolved_filters": route_dict.get("resolved_filters"),
        "requested_columns": route_dict.get("requested_columns"),
        "retrieval_sources": route_dict.get("retrieval_sources"),
        "validation_actions": route_dict.get("validation_actions"),
        "evidence_type": evidence.get("type") if isinstance(evidence, dict) else None,
        "evidence_count": evidence_count,
    }
