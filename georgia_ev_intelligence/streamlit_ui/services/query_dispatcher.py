"""QueryDispatcher — fan a single user question out to chat + map services.

Keeps the orchestration logic out of app.py so the layout file stays focused
on composition.
"""
from __future__ import annotations

from typing import Callable, Optional

from georgia_ev_intelligence.runtime_pipeline.debug_trace import (
    get_default_writer,
    session,
)
from georgia_ev_intelligence.shared.config import settings

from ..models.chat import ChatMemory
from .interfaces import DispatchResult, IChatService, IMapDataService


class QueryDispatcher:
    def __init__(self, chat_service: IChatService, map_service: IMapDataService) -> None:
        self._chat_service = chat_service
        self._map_service = map_service

    def dispatch(
        self,
        query: str,
        chat_memory: Optional[ChatMemory] = None,
        on_step: Optional[Callable[[str], None]] = None,
    ) -> DispatchResult:
        query = (query or "").strip()

        # Per-question debug tracing is opened here, at the single UI entry point,
        # so direct unit/batch callers of ChatService never write a trace file.
        if not settings.DEBUG_TRACE_ENABLED:
            return self._dispatch(query, chat_memory, on_step)

        with session(query, get_default_writer()) as recorder:
            result = self._dispatch(query, chat_memory, on_step)
            recorder.finalize(result.chat)
            return result

    def _dispatch(
        self,
        query: str,
        chat_memory: Optional[ChatMemory],
        on_step: Optional[Callable[[str], None]],
    ) -> DispatchResult:
        chat = self._chat_service.answer(query, chat_memory=chat_memory, on_step=on_step)
        map_result = self._map_service.locate(chat.effective_query or query)
        return DispatchResult(query=query, chat=chat, map=map_result)
