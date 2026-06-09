"""Pydantic v2 schemas for the route generation layer.

Three trust tiers live here:

* ``RawRoute`` / ``RawFilter`` — the *untrusted* proposal produced by the
  rule-based pre-router or the LLM router. Validated structurally only.
* ``FinalRoute`` — the *trusted* handoff contract emitted by the validator and
  consumed by the downstream execution branch.
* ``ClarificationRequest`` / ``PendingRouteState`` — the clarification loop.

Enum fields use ``str``-based enums, so ``model_dump(mode="json")`` and
``json.dumps`` both serialize them as plain strings (e.g. ``"structured_sql"``).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# List-typed fields that the LLM frequently emits as ``null`` instead of ``[]``.
# Pydantic's ``default_factory`` only applies when a key is ABSENT, not when it is
# explicitly ``None`` — so an emitted ``"sort_by": null`` would otherwise raise
# ``ValidationError: sort_by Input should be a valid list``. We coerce ``None`` ->
# ``[]`` (and a bare string -> one-element list) BEFORE validation.
_LIST_FIELDS = (
    "entities",
    "raw_filters",
    "requested_columns",
    "group_by",
    "sort_by",
    "missing_fields",
    "search_filters",
    "retrieval_sources",
    "validation_actions",
)
# Fields above whose elements are plain strings (so a bare string is wrappable).
# ``search_filters`` is intentionally excluded — its elements are objects.
_STR_LIST_FIELDS = (
    "entities",
    "requested_columns",
    "group_by",
    "sort_by",
    "missing_fields",
    "retrieval_sources",
    "validation_actions",
)


def _coerce_list_fields(data: Any) -> Any:
    """Normalize list-typed fields so ``None``/scalars never break validation."""
    if not isinstance(data, dict):
        return data
    for key in _LIST_FIELDS:
        if data.get(key) is None and key in data:
            data[key] = []
    for key in _STR_LIST_FIELDS:
        value = data.get(key)
        if isinstance(value, str):
            data[key] = [value] if value.strip() else []
    return data


# ---------------------------------------------------------------------------
# Control vocabularies (fixed system labels — never KB data values)
# ---------------------------------------------------------------------------
class RouteName(str, Enum):
    """The fixed set of control routes. The LLM may not invent new ones."""

    no_retrieval = "no_retrieval"
    exact_lookup = "exact_lookup"
    keyword_search = "keyword_search"
    structured_sql = "structured_sql"
    geo_search = "geo_search"
    vector_search = "vector_search"
    hybrid_search = "hybrid_search"
    disruption_analysis = "disruption_analysis"
    clarification_needed = "clarification_needed"
    out_of_domain = "out_of_domain"


class Operation(str, Enum):
    """Allowed route-level operations (README §34). Control labels, not KB values."""

    direct_response = "direct_response"
    lookup_entity = "lookup_entity"
    list_records = "list_records"
    count_records = "count_records"
    aggregate_records = "aggregate_records"
    group_records = "group_records"
    keyword_search = "keyword_search"
    semantic_search = "semantic_search"
    hybrid_search = "hybrid_search"
    nearby_search = "nearby_search"
    distance_search = "distance_search"
    find_alternatives = "find_alternatives"
    risk_analysis = "risk_analysis"
    ask_clarification = "ask_clarification"
    reject_out_of_domain = "reject_out_of_domain"


class FilterOperator(str, Enum):
    """Comparison operator attached to a resolved filter."""

    EQUALS = "EQUALS"
    IN = "IN"                   # OR semantics across several *exact* values
    CONTAINS = "CONTAINS"       # substring / partial match (single value)
    OR_CONTAINS = "OR_CONTAINS"  # substring match against ANY of several values
    OR_EQUALS = "OR_EQUALS"      # exact match against ANY of several values
    GT = "GT"
    LT = "LT"
    GTE = "GTE"
    LTE = "LTE"
    BETWEEN = "BETWEEN"
    CLARIFICATION_NEEDED = "CLARIFICATION_NEEDED"


class FilterStatus(str, Enum):
    resolved = "resolved"
    unresolved = "unresolved"
    clarification_needed = "clarification_needed"


# ---------------------------------------------------------------------------
# Raw (untrusted) proposal
# ---------------------------------------------------------------------------
class RawFilter(BaseModel):
    """A filter mention exactly as produced by the LLM / pre-router.

    ``raw_value`` is preserved verbatim (string, list, etc.) so the validator
    can resolve it against dynamic metadata. ``field_hint`` is a user-language
    hint (e.g. "tier", "facility") mapped to a real column later.
    """

    field_hint: str | None = None
    raw_value: Any = None
    source_text: str | None = None
    source: str = "llm"


class RawRoute(BaseModel):
    """Untrusted route proposal. Structurally validated, not yet trusted."""

    route: RouteName
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    entities: list[str] = Field(default_factory=list)
    raw_filters: list[RawFilter] = Field(default_factory=list)
    operation: str | None = None
    requested_columns: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    sort_by: list[str] = Field(default_factory=list)
    limit: int | None = None
    query_focus: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_lists(cls, data: Any) -> Any:
        return _coerce_list_fields(data)


# ---------------------------------------------------------------------------
# Resolved filter (validator output, also embedded in resolved_filters)
# ---------------------------------------------------------------------------
class ResolvedFilter(BaseModel):
    """A filter after value resolution against dynamic metadata."""

    field_hint: str | None = None
    field: str | None = None                       # resolved real column name
    raw_value: Any = None
    operator: FilterOperator | None = None
    value: Any = None                              # canonical value or list[str]
    status: FilterStatus = FilterStatus.unresolved
    candidates: list[str] = Field(default_factory=list)  # suggestions when unresolved


class SearchFilter(BaseModel):
    """A filter whose target column is uncertain — KB-free, candidates by name only.

    ``field_candidates`` are *schema field names* (never KB data values), ordered most-
    to least-likely. The executor will try them per ``fallback_policy``; until then it is
    audit/forward-looking. Produced by the validator when the LLM's ``field_hint`` is
    unmappable or risky (e.g. a tier/OEM classification dropped on the wrong field).
    """

    raw_value: Any = None
    field_candidates: list[str] = Field(default_factory=list)
    operator: str = "CONTAINS"
    fallback_policy: str = "try_in_order_then_multi_field"
    source_text: str | None = None


# ---------------------------------------------------------------------------
# Final (trusted) handoff contract
# ---------------------------------------------------------------------------
class FinalRoute(BaseModel):
    """The trusted route contract handed to the execution branch.

    ``resolved_filters`` maps real column -> ``{"operator": ..., "value": ...}``
    so the executor has both the comparison and the canonical value(s).
    """

    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_lists(cls, data: Any) -> Any:
        return _coerce_list_fields(data)

    question: str
    route: RouteName
    confidence: float = Field(ge=0.0, le=1.0)

    operation: str | None = None
    entities: list[str] = Field(default_factory=list)

    raw_filters: list[RawFilter] = Field(default_factory=list)
    resolved_filters: dict[str, Any] = Field(default_factory=dict)
    search_filters: list[SearchFilter] = Field(default_factory=list)

    requested_columns: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    sort_by: list[str] = Field(default_factory=list)
    limit: int | None = None

    query_focus: str | None = None

    needs_kb_access: bool
    needs_document_retrieval: bool
    retrieval_sources: list[str] = Field(default_factory=list)

    missing_fields: list[str] = Field(default_factory=list)
    validation_actions: list[str] = Field(default_factory=list)

    validation_status: Literal["valid", "clarification_needed", "invalid"]
    route_source: Literal[
        "pre_router_validated",
        "llm_router_validated",
        "validator_corrected",
        "clarification_merged",
        "fallback",
    ]

    clarification: dict[str, Any] | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Clarification loop
# ---------------------------------------------------------------------------
class PendingRouteState(BaseModel):
    """State carried between a clarification request and the user's reply.

    The user's clarification fills missing fields in this state; it is *not* a
    new standalone question.
    """

    session_id: str | None = None
    original_question: str
    proposed_route: RouteName | None = None
    raw_route: RawRoute | None = None
    known_fields: dict[str, Any] = Field(default_factory=dict)
    resolved_filters: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    candidates: dict[str, list[str]] = Field(default_factory=dict)
    route_reason: str = ""


class ClarificationRequest(BaseModel):
    needed: bool = True
    missing_fields: list[str] = Field(default_factory=list)
    question_to_user: str
    pending_route_state: PendingRouteState | dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API request / response models (README §37)
# ---------------------------------------------------------------------------
class RouteRequest(BaseModel):
    question: str
    session_id: str | None = None


class ClarifyRequest(BaseModel):
    """Either resolve the pending state from ``session_id`` (preferred) or pass
    an explicit ``pending_route_state`` (README §37 contract)."""

    session_id: str | None = None
    user_clarification: str
    pending_route_state: dict[str, Any] | None = None


class RouteResponse(BaseModel):
    status: Literal["routed", "needs_clarification"]
    session_id: str
    final_route: FinalRoute | None = None
    clarification: ClarificationRequest | None = None
