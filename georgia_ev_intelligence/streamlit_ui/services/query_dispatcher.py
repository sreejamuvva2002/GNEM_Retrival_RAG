"""QueryDispatcher — fan a single user question out to chat + map services.

Keeps the orchestration logic out of app.py so the layout file stays focused
on composition.
"""
from __future__ import annotations

from .interfaces import DispatchResult, IChatService, IMapDataService


class QueryDispatcher:
    def __init__(self, chat_service: IChatService, map_service: IMapDataService) -> None:
        self._chat_service = chat_service
        self._map_service = map_service

    def dispatch(self, query: str) -> DispatchResult:
        query = (query or "").strip()
        chat = self._chat_service.answer(query)
        map_result = self._map_service.locate(query)
        return DispatchResult(query=query, chat=chat, map=map_result)
