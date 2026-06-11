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
    # Company rows the answer actually drew on, ready for the map (each carries
    # company/latitude/longitude). Populated by route-backed services whose
    # evidence rows already include coordinates; empty for services that rely on
    # the separate map pipeline + cited-company filter.
    map_records: List[Dict[str, Any]] = field(default_factory=list)
    # Provenance for the answer ("sources"). For SQL-backed routes the records
    # the answer is grounded in live here (one dict per KB row), and the executed
    # query is recorded separately so the UI can show evidence (records) and
    # method (SQL) as distinct things — see RouteChatService._provenance_*.
    #   evidence_kind: "records" (per-company rows) | "groups" (aggregate rows)
    #                  | "count" (a single number) | "" (none / document route).
    #   evidence_rows: the records or group rows backing the answer.
    #   sql_queries:   [{"label": ..., "sql": ...}] executed to retrieve them.
    evidence_kind: str = ""
    evidence_rows: List[Dict[str, Any]] = field(default_factory=list)
    sql_queries: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class DispatchResult:
    """Combined chat + map output for a single submitted query."""

    query: str
    chat: ChatResult
    map: MapResult


class IChatService(Protocol):
    def answer(
        self, query: str, history: list[tuple[str, str]] | None = None, on_step: Optional[Callable[[str], None]] = None
    ) -> ChatResult: ...


class IMapDataService(Protocol):
    def locate(self, query: str) -> MapResult: ...
