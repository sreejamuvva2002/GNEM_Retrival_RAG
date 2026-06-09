"""QueryDispatcher — fan a single user question out to chat + map services.

Keeps the orchestration logic out of app.py so the layout file stays focused
on composition.
"""
from __future__ import annotations

from typing import Callable, Optional

from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext

from ..models.map import MapContext, MapResult
from ..spatial.query_planner import QueryPlanner
from .interfaces import ChatResult, DispatchResult, IChatService, IMapDataService


class QueryDispatcher:
    def __init__(
        self,
        chat_service: IChatService,
        map_service: IMapDataService,
        query_router: QueryPlanner | None = None,
    ) -> None:
        self._chat_service = chat_service
        self._map_service = map_service
        self._query_router = query_router or QueryPlanner()

    def dispatch(
        self,
        query: str,
        history: list[tuple[str, str]] | None = None,
        previous_contexts: list[ParentContext] | None = None,
        on_step: Optional[Callable[[str], None]] = None,
    ) -> DispatchResult:
        query = (query or "").strip()
        route = self._query_router.plan(query)
        if route.get("classification") == "NO_RETRIEVAL":
            hints = route.get("hints", {}) or {}
            if on_step is not None:
                try:
                    on_step("generation")
                except Exception:
                    pass
            return DispatchResult(
                query=query,
                chat=ChatResult(
                    answer=str(hints.get("direct_answer") or ""),
                    parent_contexts=[],
                    trace={
                        "route": "NO_RETRIEVAL",
                        "route_reason": str(hints.get("route_reason") or "no_retrieval"),
                    },
                ),
                map=MapResult(
                    records=[],
                    context=MapContext(map_mode="no_retrieval"),
                ),
            )

        chat = self._chat_service.answer(
            query,
            history=history,
            previous_contexts=previous_contexts,
            on_step=on_step,
        )
        map_result = self._map_service.locate(query)
        return DispatchResult(query=query, chat=chat, map=map_result)
