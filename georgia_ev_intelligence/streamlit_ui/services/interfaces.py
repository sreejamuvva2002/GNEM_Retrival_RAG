"""Service protocols — what each consumer of the chat / map flow depends on.

Keeping these as Protocols (not concrete classes) lets components stay
free of implementation details, satisfies DIP, and makes testing trivial
(any mock that satisfies the protocol works).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext

from ..models.map import MapResult


@dataclass
class ChatResult:
    """Output of one chat round-trip.

    `parent_contexts` is filtered to only the parents whose company was named
    in the LLM's `used_companies` list. `warn` carries a short, user-visible
    notice when JSON parsing falls back (UI may show "showing all sources").
    """

    answer: str
    parent_contexts: List[ParentContext]
    trace: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    warn: str = ""


@dataclass
class DispatchResult:
    """Combined chat + map output for a single submitted query."""

    query: str
    chat: ChatResult
    map: MapResult


class IChatService(Protocol):
    def answer(
        self, query: str, on_step: Optional[Callable[[str], None]] = None
    ) -> ChatResult: ...


class IMapDataService(Protocol):
    def locate(self, query: str) -> MapResult: ...
