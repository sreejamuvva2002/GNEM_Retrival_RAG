"""QueryDispatcher — fan a single user question out to chat + map services.

Keeps the orchestration logic out of app.py so the layout file stays focused
on composition.
"""
from __future__ import annotations

from typing import Callable, Optional

from ..models.map import MapContext, MapResult
from .interfaces import DispatchResult, IChatService, IMapDataService


class QueryDispatcher:
    def __init__(self, chat_service: IChatService, map_service: IMapDataService) -> None:
        self._chat_service = chat_service
        self._map_service = map_service

    def dispatch(
        self,
        query: str,
        history: list[tuple[str, str]] | None = None,
        on_step: Optional[Callable[[str], None]] = None,
    ) -> DispatchResult:
        query = (query or "").strip()
        chat = self._chat_service.answer(query, history=history, on_step=on_step)
        if str(chat.trace.get("route") or "") == "no_retrieval":
            return DispatchResult(
                query=query,
                chat=chat,
                map=MapResult(records=[], context=MapContext(map_mode="no_retrieval")),
            )
        map_result = self._map_service.locate(query)
        return DispatchResult(query=query, chat=chat, map=map_result)
