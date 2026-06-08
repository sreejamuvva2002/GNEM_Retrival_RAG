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

import re
from typing import Any, Callable

from georgia_ev_intelligence.route_execution import execute_route
from georgia_ev_intelligence.route_execution.schemas import STATUS_FAILED

from .interfaces import ChatResult, IChatService

_CONTEXT_GEO_RE = re.compile(
    r"\b(distance|near|nearby|nearest|closest|within)\b.*\b(these|those|them|listed|above|previous)\b|"
    r"\b(these|those|them|listed|above|previous)\b.*\b(distance|near|nearby|nearest|closest|within)\b",
    re.IGNORECASE,
)
_NUMBERED_COMPANY_RE = re.compile(
    r"^\s*\d+\.\s+(.+?)(?=\s+-\s+|\s*:\s+\d+(?:\.\d+)?\s+(?:mi|mile)|\s*$)",
    re.MULTILINE | re.IGNORECASE,
)
_DISTANCE_RE = re.compile(r"\bdistance\b", re.IGNORECASE)
_NEAREST_RE = re.compile(r"\b(nearest|closest)\b", re.IGNORECASE)
_CENTER_HEADING_RE = re.compile(
    r"^(?:Distances\s+to|Companies\s+within\s+\d+(?:\.\d+)?\s+miles\s+of)\s+(.+?):\s*$",
    re.MULTILINE | re.IGNORECASE,
)


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

    def answer(self, query: str, history: list[tuple[str, str]] | None = None, on_step: Callable[[str], None] | None = None) -> ChatResult:
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
            final_route = self._route_service_lazy().route(query, history=history)
            route_dict = final_route.model_dump(mode="json")
            context_entities = _context_entities_for_distance(query, history)
            context_center = _context_center_for_geo(history)
            route_entities = [
                str(entity).strip()
                for entity in (route_dict.get("entities") or [])
                if str(entity).strip()
            ]
            center = route_entities[-1] if route_entities else context_center
            if context_entities and center:
                route_dict["route"] = "geo_search"
                route_dict["validation_status"] = "valid"
                route_dict["missing_fields"] = []
                route_dict["clarification"] = None
                route_dict["needs_kb_access"] = True
                route_dict["needs_document_retrieval"] = False
                route_dict["retrieval_sources"] = ["structured_db"]
                route_dict["entities"] = [center]
                route_dict["context_entities"] = context_entities
                route_dict["operation"] = (
                    "distance_search"
                    if _DISTANCE_RE.search(query) or _NEAREST_RE.search(query)
                    else "nearby_search"
                )
                if _NEAREST_RE.search(query):
                    route_dict["limit"] = 1
                removed_filters = route_dict.get("resolved_filters") or {}
                route_dict["resolved_filters"] = {}
                route_dict.setdefault("validation_actions", []).append(
                    f"resolved {len(context_entities)} geo targets from conversation history"
                )
                if removed_filters:
                    route_dict["validation_actions"].append(
                        "removed center company from structured filters"
                    )
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

        evidence = getattr(result, "evidence", None)
        evidence_kind, evidence_rows = _provenance_rows_from_evidence(evidence)
        return ChatResult(
            answer=result.answer or "(no answer returned)",
            parent_contexts=[],
            trace=trace,
            map_records=_map_records_from_evidence(evidence),
            evidence_kind=evidence_kind,
            evidence_rows=evidence_rows,
            sql_queries=_sql_queries_from_evidence(evidence),
        )


def _coord(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # reject NaN


def _map_records_from_evidence(evidence: Any) -> list[dict[str, Any]]:
    """Build map_view records from the geocoded rows in the executor evidence.

    The route executor returns every company the answer is grounded in under
    ``evidence["rows"]`` (each row carries ``latitude``/``longitude``). We map
    those straight to the marker shape map_view expects so the map shows exactly
    the companies in the answer — no separate spatial pipeline, no re-filtering.
    Rows without usable coordinates are skipped (they can't be placed).
    """
    if not isinstance(evidence, dict):
        return []
    rows = evidence.get("rows")
    if not isinstance(rows, (list, tuple)):
        return []

    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        lat = _coord(row.get("latitude"))
        lon = _coord(row.get("longitude"))
        if lat is None or lon is None:
            continue
        records.append(
            {
                "company": row.get("company"),
                "latitude": lat,
                "longitude": lon,
                "address": row.get("updated_location"),
                "product_service": row.get("product_service"),
            }
        )
    return records


def _provenance_rows_from_evidence(evidence: Any) -> tuple[str, list[dict[str, Any]]]:
    """Classify the answer's provenance and return its backing rows.

    The "sources" of a SQL-backed answer are the KB rows the query returned —
    the structured analogue of retrieved chunks. We surface them so the UI can
    show one source card per record. Counts have no rows (just a number);
    aggregates expose their group rows; other evidence types (e.g. document
    chunks) are left to the parent-context path.
    """
    if not isinstance(evidence, dict):
        return "", []
    etype = str(evidence.get("type") or "")
    if etype in ("structured_rows", "geo_results"):
        rows = evidence.get("rows")
        return "records", [r for r in (rows or []) if isinstance(r, dict)]
    if etype in ("group_counts", "group_aggregates"):
        groups = evidence.get("groups")
        return "groups", [g for g in (groups or []) if isinstance(g, dict)]
    if etype == "count":
        return "count", []
    return "", []


def _sql_queries_from_evidence(evidence: Any) -> list[dict[str, str]]:
    """Collect the executed query/queries as readable provenance (not a source).

    structured_sql records one query at the top level; geo_search records one or
    more under ``sql_commands``. We return the display form (literals inlined) so
    the UI can show *how* the records were retrieved, kept distinct from the
    records themselves.
    """
    if not isinstance(evidence, dict):
        return []
    queries: list[dict[str, str]] = []
    top_sql = evidence.get("sql_display") or evidence.get("sql")
    if top_sql:
        # spatial_operation is descriptive ("ST_DWithin_company"); the structured
        # evidence types ("structured_rows", "count", …) are not, so label those
        # generically.
        label = str(evidence.get("spatial_operation") or "SQL query")
        queries.append({"label": label, "sql": str(top_sql)})
    for command in evidence.get("sql_commands") or []:
        if not isinstance(command, dict):
            continue
        sql = command.get("sql_display") or command.get("sql")
        if sql:
            queries.append({"label": str(command.get("label") or "Query"), "sql": str(sql)})
    return queries


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
        "context_entities": route_dict.get("context_entities"),
        "evidence_type": evidence.get("type") if isinstance(evidence, dict) else None,
        "evidence_count": evidence_count,
    }


def _context_entities_for_distance(
    query: str,
    history: list[tuple[str, str]] | None,
) -> list[str]:
    """Extract prior numbered result companies for an anaphoric geo follow-up."""
    if not history or not _CONTEXT_GEO_RE.search(query or ""):
        return []

    for role, content in reversed(history):
        if str(role).casefold() != "assistant":
            continue
        companies = [
            match.group(1).strip()
            for match in _NUMBERED_COMPANY_RE.finditer(str(content or ""))
            if match.group(1).strip()
        ]
        if companies:
            return list(dict.fromkeys(companies))
    return []


def _context_center_for_geo(
    history: list[tuple[str, str]] | None,
) -> str | None:
    """Resolve the latest named center from a prior distance/nearby answer."""
    for role, content in reversed(history or []):
        if str(role).casefold() != "assistant":
            continue
        match = _CENTER_HEADING_RE.search(str(content or ""))
        if match:
            return match.group(1).strip()
    return None
