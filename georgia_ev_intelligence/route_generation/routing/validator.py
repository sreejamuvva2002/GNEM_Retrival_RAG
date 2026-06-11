"""Router validator — turns an untrusted ``RawRoute`` into a trusted ``FinalRoute``.

Design stance (README §25-27): validate *structure and executability* only — the
validator is fully **KB-free**. It never reads KB data values: no canonicalization,
no fuzzy matching, no metadata candidate lists, and no quoted-value field repair.
Every filter value is preserved *verbatim* (text-normalized only); category/tier
values use exact matching while free-text values use safe substring matching.
Repeated filters merge into OR groups; numeric employment thresholds, grouped
employment aggregation, and ranking come from the question text; output columns
are separated from filters; and a value the LLM dropped on the wrong field is
rescued by question wording (not KB). Clarification is reserved for genuinely
missing information.

Sequence:

1. Parse RawRoute (already pydantic-valid; list fields coerced from null -> []).
2. Deterministic route correction by question signals (incl. multi-entity and
   numeric-comparison overrides so multi-company questions never stay exact_lookup).
3. Build resolved filters: map hints -> columns, OR-merge repeats, preserve every
   value with exact category matching or free-text substring matching, cross-field
   rescue (wording), numeric employment.
   A classification value on the wrong field is reverse-rescued onto ``category``;
   a value with an unmappable hint becomes a ``SearchFilter`` (candidate columns by
   name, never KB values) instead of being silently dropped.
4. Separate requested output columns from filters.
5. Hybrid upgrade when structured filters meet web/document-evidence wording.
6. Relaxed required-field check + confidence.
7. Emit FinalRoute (with ``validation_actions`` recording every change and
   ``retrieval_sources``), or a clarification route only when info is truly missing.
"""
from __future__ import annotations

import re
from typing import Any

from ..config import ROUTER_CONFIDENCE_THRESHOLD
from ..metadata.provider import MetadataProvider
from ..schemas import (
    ClarificationRequest,
    FilterOperator,
    FinalRoute,
    Operation,
    PendingRouteState,
    RawRoute,
    RouteName,
    SearchFilter,
)
from ..utils.text_normalization import normalize_for_field
from .field_mapping import (
    AGGREGATE_SIGNALS,
    COMPARISON_SIGNALS,
    DISRUPTION_SIGNALS,
    LIST_SIGNALS,
    MULTI_ENTITY_SIGNALS,
    PROXIMITY_SIGNALS,
    TIER_FIELD_CANDIDATES,
    WEB_EVIDENCE_SIGNALS,
    detect_output_columns,
    is_field_name_echo,
    looks_like_category,
    matches_any,
    normalize_operation,
    parse_employment_comparisons,
    parse_ranking,
    rescue_text_field,
    route_needs_flags,
    route_retrieval_sources,
)

# Routes that never require KB fields and never clarify.
_NO_FIELD_ROUTES = {RouteName.no_retrieval, RouteName.out_of_domain}
# "Basic" routes the deterministic correction may upgrade (never downgrade a
# richer route like hybrid_search or structured_sql).
_CORRECTABLE = {
    RouteName.vector_search,
    RouteName.keyword_search,
    RouteName.exact_lookup,
    RouteName.no_retrieval,
    RouteName.out_of_domain,
}
_GEO_CORRECTABLE = {*_CORRECTABLE, RouteName.structured_sql}
_STRUCTURED_CORRECTABLE = {*_CORRECTABLE, RouteName.geo_search}
# Routes that may be upgraded to hybrid_search when structured filters are paired
# with web/document-evidence wording (keyword/vector keep their own distinction).
_HYBRID_UPGRADABLE = {
    RouteName.structured_sql,
    RouteName.vector_search,
    RouteName.keyword_search,
    RouteName.exact_lookup,
}
_COORD_RE = re.compile(r"-?\d{1,2}\.\d+\s*,\s*-?\d{1,3}\.\d+")
_PROXIMITY_PLACE_RE = re.compile(
    r"(?:\bnear\b|\baround\b|\bclosest\s+to\b|"
    r"\bwithin\s+\d+(?:\.\d+)?\s*(?:km|kilometers?|miles?|mi)\s+of\b)"
    r"\s+[a-z][a-z .'-]+(?:[?.!,]|$)",
    re.IGNORECASE,
)
_GEO_ANCHOR_WORDS = {
    "near", "nearby", "closest", "county", "counties", "around", "in", "distance",
}
_MAP_OR_SPATIAL_SIGNALS = {"map", "geospatial", "spatial", "county", "counties", "coordinates"}
# The exact-classification field the LLM most often over-assigns; the only field
# eligible for cross-field rescue.
_RESCUE_SOURCE_FIELD = "category"
# Values that carry no filtering information.
_EMPTY_VALUES = {"", "unknown", "none", "n/a", "na", "null"}
_OR_SPLIT_RE = re.compile(r"\bor\b", re.IGNORECASE)
_GENERIC_ROLE_VALUES = {"supplier", "suppliers"}
_GENERIC_ENTITY_SUFFIX_RE = re.compile(
    r"\s+(?:suppliers?|companies|firms|manufacturers?)(?:\s+only)?\s*$",
    re.IGNORECASE,
)
_TOTAL_EMPLOYMENT_RE = re.compile(
    r"\b(total|sum)\b.{0,50}\bemploy\w*|\bemploy\w*.{0,50}\b(total|sum)\b",
    re.IGNORECASE,
)

_MISSING_QUESTION = {
    "query_focus": "What topic should I focus on for the search?",
    "location": "Which location, city, or county should I use as the center?",
    "entity": "Which company or entity should I look up?",
    "filter_or_aggregate": "Which field should I filter, list, or group by?",
    "structured_or_geo_signal": "Should I also filter by a field or a location?",
    "intent": "Could you rephrase what you're looking for?",
}


class RouteValidator:
    def __init__(self, provider: MetadataProvider, confidence_threshold: float | None = None) -> None:
        self._provider = provider
        self._threshold = (
            confidence_threshold if confidence_threshold is not None else ROUTER_CONFIDENCE_THRESHOLD
        )

    def validate(
        self,
        raw_route: RawRoute,
        normalized: dict,
        route_source: str = "llm_router_validated",
    ) -> FinalRoute:
        question = normalized.get("normalized") or normalized.get("original") or ""
        lower = normalized.get("lowercase", "")
        actions: list[str] = []

        # deterministic route correction
        route = raw_route.route
        source = route_source
        corrected, correction_reason = self._correct_route(route, lower)
        if corrected != route:
            actions.append(
                f"changed route {route.value} -> {corrected.value} ({correction_reason})"
            )
            route = corrected
            source = "validator_corrected"

        # requested columns + resolved filters (OR-merge, preserve verbatim, rescue).
        # ``search_filters`` collects values whose target column is uncertain.
        requested_columns = self._requested_columns(raw_route, lower)
        resolved_filters, search_filters = self._resolve_filters(raw_route, lower, actions)
        if route is RouteName.geo_search and raw_route.entities:
            self._remove_geo_center_filters(
                resolved_filters,
                raw_route.entities,
                lower,
                actions,
            )

        # numeric employment threshold (KB-free) overrides any CONTAINS guess
        employment = self._employment_filter(lower)
        if employment is not None and "employment" in self._provider.get_allowed_fields():
            resolved_filters["employment"] = employment

        # ranking -> sort_by / limit (prefer deterministic parse over LLM output)
        parsed_sort, parsed_limit = parse_ranking(lower)
        sort_by = parsed_sort or list(raw_route.sort_by)
        limit = parsed_limit if parsed_limit is not None else raw_route.limit
        group_by = self._group_by(raw_route.group_by, lower, actions)

        # Hybrid upgrade: structured filters AND web/document-evidence wording ->
        # hybrid_search (structured DB + document chunks), never plain vector_search.
        structured_signal = bool(resolved_filters) or bool(requested_columns)
        if (
            route in _HYBRID_UPGRADABLE
            and structured_signal
            and matches_any(lower, WEB_EVIDENCE_SIGNALS)
        ):
            actions.append(
                f"changed route {route.value} -> hybrid_search "
                "(structured filters plus web/document evidence)"
            )
            route = RouteName.hybrid_search
            source = "validator_corrected"

        operation = normalize_operation(raw_route.operation, route.value)
        if (
            route is RouteName.structured_sql
            and _TOTAL_EMPLOYMENT_RE.search(lower)
            and group_by
            and operation != Operation.aggregate_records.value
        ):
            operation = Operation.aggregate_records.value
            actions.append("changed operation to aggregate_records (total employment by group)")
            source = "validator_corrected"
        query_focus = raw_route.query_focus

        # relaxed required-field check
        missing = self._missing_required(
            route, raw_route, resolved_filters, query_focus, lower, requested_columns, sort_by
        )

        # confidence + clarification decision
        low_confidence = (
            raw_route.confidence < self._threshold
            and source != "validator_corrected"
            and route not in _NO_FIELD_ROUTES
        )
        needs_clarification = bool(missing) or low_confidence
        if route in _NO_FIELD_ROUTES:
            needs_clarification = False

        if needs_clarification:
            if low_confidence and not missing:
                missing = ["intent"]
            return self._clarification_route(
                raw_route, route, question, resolved_filters, search_filters, missing, source, actions
            )

        # valid FinalRoute
        needs_kb, needs_doc = route_needs_flags(route.value)
        return FinalRoute(
            question=question,
            route=route,
            confidence=raw_route.confidence,
            operation=operation,
            entities=raw_route.entities,
            raw_filters=raw_route.raw_filters,
            resolved_filters=resolved_filters,
            search_filters=search_filters,
            requested_columns=requested_columns,
            group_by=group_by,
            sort_by=sort_by,
            limit=limit,
            query_focus=query_focus,
            needs_kb_access=needs_kb,
            needs_document_retrieval=needs_doc,
            retrieval_sources=route_retrieval_sources(needs_kb, needs_doc),
            missing_fields=[],
            validation_status="valid",
            route_source=source,
            clarification=None,
            validation_actions=actions,
            reason=raw_route.reason or f"Validated as {route.value}.",
        )

    # -- field mapping ------------------------------------------------------
    def _map_field(self, field_hint: str | None) -> str | None:
        if not field_hint:
            return None
        field = self._provider.resolve_field_alias(field_hint)
        if field is None or field not in self._provider.get_allowed_fields():
            return None
        return field

    # -- route correction ---------------------------------------------------
    def _correct_route(self, route: RouteName, lower: str) -> tuple[RouteName, str]:
        """Return ``(corrected_route, reason)``; reason is "" when unchanged."""
        if matches_any(lower, DISRUPTION_SIGNALS) and route is not RouteName.disruption_analysis:
            return RouteName.disruption_analysis, "risk/dependency/alternatives wording"
        if matches_any(lower, AGGREGATE_SIGNALS) and route in _STRUCTURED_CORRECTABLE:
            return RouteName.structured_sql, "aggregate/count wording"
        if matches_any(lower, COMPARISON_SIGNALS) and route in _STRUCTURED_CORRECTABLE:
            return RouteName.structured_sql, "numeric-comparison/superlative wording"
        # Geospatial intent is broader than radius/proximity: map requests and
        # county containment should also reach the PostGIS executor.
        if (
            (matches_any(lower, PROXIMITY_SIGNALS) or matches_any(lower, _MAP_OR_SPATIAL_SIGNALS) or _COORD_RE.search(lower))
            and not matches_any(lower, AGGREGATE_SIGNALS)
            and not matches_any(lower, COMPARISON_SIGNALS)
            and route in _GEO_CORRECTABLE
        ):
            return RouteName.geo_search, "geospatial wording"
        # Multi-company questions must never stay exact_lookup (single-entity only).
        if route is RouteName.exact_lookup and matches_any(lower, MULTI_ENTITY_SIGNALS):
            return RouteName.structured_sql, "multi-entity question"
        if matches_any(lower, LIST_SIGNALS) and route in _CORRECTABLE:
            return RouteName.structured_sql, "list wording"
        return route, ""

    # -- requested output columns ------------------------------------------
    def _requested_columns(self, raw_route: RawRoute, lower: str) -> list[str]:
        """Columns the user asked to SHOW — kept out of the filter set."""
        allowed = set(self._provider.get_allowed_fields())
        cols: list[str] = []
        for raw_col in raw_route.requested_columns:
            field = self._map_field(raw_col)
            if field is None and raw_col in allowed:
                field = raw_col
            if field:
                cols.append(field)
        for field in detect_output_columns(lower):
            if field in allowed:
                cols.append(field)
        return list(dict.fromkeys(cols))

    # -- filter resolution (OR-merge + soften + rescue) ---------------------
    def _resolve_filters(
        self, raw_route: RawRoute, lower: str, actions: list[str]
    ) -> tuple[dict[str, dict], list[SearchFilter]]:
        """Return ``(resolved_filters, search_filters)`` — fully KB-free.

        Repeated filters for one field OR-merge instead of overwriting. A value the
        LLM dropped on the wrong field is rescued by wording: a tier/OEM-like value
        on a non-category field is reverse-rescued onto ``category`` (and mirrored as
        a ``SearchFilter`` for the future executor); a value with an unmappable hint
        becomes a ``SearchFilter`` rather than being silently dropped.
        """
        grouped: dict[str, list[Any]] = {}
        search_filters: list[SearchFilter] = []
        for raw_filter in raw_route.raw_filters:
            field = self._map_field(raw_filter.field_hint)
            if field is None:
                # Known-but-non-filterable column (alias resolves) -> drop quietly.
                # Genuinely unknown / missing hint with a real value -> search_filter.
                if self._is_known_column(raw_filter.field_hint):
                    continue
                sf = self._build_search_filter(raw_filter, lower)
                if sf is not None:
                    search_filters.append(sf)
                    actions.append(
                        f"created search_filter for '{raw_filter.raw_value}' "
                        f"(uncertain field; candidates {sf.field_candidates})"
                    )
                continue
            if (
                field == "ev_supply_chain_role"
                and str(raw_filter.raw_value or "").strip().casefold() in _GENERIC_ROLE_VALUES
            ):
                actions.append(
                    f"dropped generic '{raw_filter.raw_value}' role filter "
                    "(supplier is the requested entity type, not a supply-chain role)"
                )
                continue
            field, sf, action = self._route_filter_field(field, raw_filter, lower)
            if sf is not None:
                search_filters.append(sf)
            if action:
                actions.append(action)
            raw_value = raw_filter.raw_value
            if field == "ev_supply_chain_role":
                raw_value = self._strip_generic_entity_suffix(raw_value)
                if raw_value != raw_filter.raw_value:
                    actions.append(
                        f"removed generic entity wording from role filter "
                        f"'{raw_filter.raw_value}' -> '{raw_value}'"
                    )
                if raw_value in (None, "", []):
                    continue
            grouped.setdefault(field, []).append(raw_value)

        resolved: dict[str, dict] = {}
        for field, raw_values in grouped.items():
            combined = self._combine_field(field, raw_values)
            if combined is None:
                continue  # only field-name echoes / empties -> output column, not a filter
            operator, value = combined
            resolved[field] = {"operator": operator, "value": value}
        return resolved, search_filters

    def _remove_geo_center_filters(
        self,
        resolved_filters: dict[str, Any],
        entities: list[str],
        lower: str,
        actions: list[str],
    ) -> None:
        """Do not reuse the named spatial center as a candidate-row filter."""
        entity_values = {
            re.sub(r"[^a-z0-9]+", " ", str(entity).casefold()).strip()
            for entity in entities
            if str(entity).strip()
        }
        explicit_relationship = any(
            token in lower
            for token in ("linked to", "supplies", "support", "customer", "primary oem", "oem")
        )
        for field in ("company", "updated_location", "primary_oems"):
            if field == "primary_oems" and explicit_relationship:
                continue
            spec = resolved_filters.get(field)
            if not isinstance(spec, dict):
                continue
            raw_values = spec.get("value")
            values = raw_values if isinstance(raw_values, (list, tuple)) else [raw_values]
            normalized = {
                re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
                for value in values
                if str(value or "").strip()
            }
            if normalized and normalized <= entity_values:
                resolved_filters.pop(field, None)
                actions.append(
                    f"removed {field} filter because it identifies the geo center"
                )

    def _strip_generic_entity_suffix(self, value: Any) -> Any:
        """Remove trailing entity nouns accidentally included in role values."""
        if isinstance(value, list):
            return [self._strip_generic_entity_suffix(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._strip_generic_entity_suffix(item) for item in value)
        if not isinstance(value, str):
            return value
        cleaned = value.strip()
        while True:
            stripped = _GENERIC_ENTITY_SUFFIX_RE.sub("", cleaned).strip()
            if stripped == cleaned:
                return stripped
            cleaned = stripped

    def _is_known_column(self, field_hint: str | None) -> bool:
        """True when a hint maps to a real KB column (even a non-filterable one)."""
        return bool(field_hint) and self._provider.resolve_field_alias(field_hint) is not None

    def _route_filter_field(
        self, field: str, raw_filter, lower: str
    ) -> tuple[str, SearchFilter | None, str]:
        """Pick the target column for a mapped filter, KB-free.

        Returns ``(target_field, search_filter_or_None, action_or_empty)``. A
        tier/OEM-like value on a non-category field is reverse-rescued onto
        ``category`` (and mirrored as a SearchFilter); a non-classification value on
        ``category`` is forward-rescued to a searchable text field by wording.
        """
        raw_value = raw_filter.raw_value
        # Reverse rescue: classification value mis-assigned to a non-category field.
        if field != _RESCUE_SOURCE_FIELD and looks_like_category(raw_value):
            sf = SearchFilter(
                raw_value=raw_value,
                field_candidates=[
                    c for c in TIER_FIELD_CANDIDATES
                    if c in self._provider.get_allowed_fields()
                ],
                source_text=getattr(raw_filter, "source_text", None),
            )
            action = (
                f"moved '{raw_value}' from {field} to category (classification wording)"
            )
            return _RESCUE_SOURCE_FIELD, sf, action
        # Forward rescue: non-classification value dropped on category.
        if field == _RESCUE_SOURCE_FIELD and not looks_like_category(raw_value):
            target = rescue_text_field(raw_value, lower)
            if target in self._provider.get_allowed_fields() and target != field:
                return target, None, f"moved '{raw_value}' from category to {target} (wording)"
        return field, None, ""

    def _build_search_filter(self, raw_filter, lower: str) -> SearchFilter | None:
        """Build a SearchFilter for a value whose target column is uncertain.

        Candidate columns are schema field NAMES (never KB values), ordered most- to
        least-likely by wording. Returns None for empty / placeholder values.
        """
        text = str(raw_filter.raw_value or "").strip()
        if not text or text.lower() in _EMPTY_VALUES:
            return None
        allowed = set(self._provider.get_allowed_fields())
        if looks_like_category(raw_filter.raw_value):
            ordered = list(TIER_FIELD_CANDIDATES)
        else:
            best = rescue_text_field(raw_filter.raw_value, lower)
            ordered = [best, "ev_supply_chain_role", "product_service", "category"]
        candidates = [c for c in dict.fromkeys(ordered) if c in allowed]
        if not candidates:
            return None
        return SearchFilter(
            raw_value=raw_filter.raw_value,
            field_candidates=candidates,
            source_text=getattr(raw_filter, "source_text", None),
        )

    def _combine_field(self, field: str, raw_values: list[Any]):
        """Combine all values for one field into ``(operator, value)`` — KB-free.

        Every value is preserved *verbatim* (text-normalized only, never matched or
        canonicalized against KB data). Category values use exact matching; other
        text values use substring matching. Repeated filters are OR-merged, so the
        last value never overwrites earlier ones.
        """
        out_values: list[Any] = []
        for raw_value in raw_values:
            # Preserve each value; drop bare field-name echoes ("EV Supply Chain
            # Role") since those mark an OUTPUT column, not a filter.
            out_values.extend(
                p for p in self._soft_split(field, raw_value)
                if not is_field_name_echo(field, p)
            )

        out_values = list(dict.fromkeys(v for v in out_values if str(v).strip()))
        if not out_values:
            return None
        if len(out_values) == 1 and field == "category":
            return FilterOperator.EQUALS.value, out_values[0]
        if len(out_values) > 1 and field == "category":
            return FilterOperator.OR_EQUALS.value, out_values
        if len(out_values) == 1:
            return FilterOperator.CONTAINS.value, out_values[0]
        return FilterOperator.OR_CONTAINS.value, out_values

    def _group_by(self, raw_group_by: list[str], lower: str, actions: list[str]) -> list[str]:
        """Normalize safe group fields and infer county for county aggregate questions."""
        groups: list[str] = []
        for raw_field in raw_group_by:
            folded = str(raw_field or "").strip().casefold()
            if folded in {"county", "counties"}:
                groups.append("county")
                continue
            field = self._map_field(raw_field)
            if field is not None:
                groups.append(field)
            elif folded:
                actions.append(f"dropped unsupported group_by field '{raw_field}'")

        if not groups and "county" in lower and _TOTAL_EMPLOYMENT_RE.search(lower):
            groups.append("county")
            actions.append("inferred group_by county from total employment question")
        return list(dict.fromkeys(groups))

    def _soft_split(self, field: str, raw_value: Any) -> list[str]:
        """Preserve an unmatched value, splitting only clear multi-value phrases."""
        if isinstance(raw_value, (list, tuple, set)):
            items = [str(v).strip() for v in raw_value if str(v).strip()]
        else:
            text = str(raw_value or "").strip()
            if not text or text.lower() in _EMPTY_VALUES:
                return []
            if _OR_SPLIT_RE.search(text):
                items = [p.strip() for p in _OR_SPLIT_RE.split(text) if p.strip()]
            elif "/" in text and all(re.search(r"[a-zA-Z]", part) for part in text.split("/")):
                items = [p.strip() for p in text.split("/") if p.strip()]
            else:
                items = [text]
        items = [i for i in items if i and i.lower() not in _EMPTY_VALUES]
        # Strip leading and trailing quotes that the LLM may have left behind
        items = [i.strip("'\"").strip() for i in items]
        items = [i for i in items if i]
        return [normalize_for_field(field, i) for i in items]

    # -- numeric employment -------------------------------------------------
    def _employment_filter(self, lower: str) -> dict | None:
        comparisons = parse_employment_comparisons(lower)
        if not comparisons:
            return None
        lows = [v for op, v in comparisons if op in ("GT", "GTE")]
        highs = [v for op, v in comparisons if op in ("LT", "LTE")]
        if lows and highs:
            return {"operator": FilterOperator.BETWEEN.value, "value": [min(lows), max(highs)]}
        operator, value = comparisons[0]
        return {"operator": operator, "value": value}

    # -- required fields (relaxed) ------------------------------------------
    def _missing_required(
        self,
        route: RouteName,
        raw_route: RawRoute,
        resolved_filters: dict,
        query_focus,
        lower: str,
        requested_columns: list[str],
        sort_by: list[str],
    ) -> list[str]:
        has_filter = bool(resolved_filters)
        has_entity = bool(raw_route.entities)
        focus = (query_focus or "").strip()
        # Any sign the question describes a structured request we can execute.
        structured_intent = (
            has_filter
            or bool(requested_columns)
            or bool(sort_by)
            or bool(raw_route.group_by)
            or matches_any(lower, AGGREGATE_SIGNALS)
            or matches_any(lower, COMPARISON_SIGNALS)
            or matches_any(lower, LIST_SIGNALS)
            or matches_any(lower, MULTI_ENTITY_SIGNALS)
        )

        if route in _NO_FIELD_ROUTES or route is RouteName.clarification_needed:
            return []
        if route is RouteName.exact_lookup:
            return [] if ("company" in resolved_filters or has_entity) else ["entity"]
        if route is RouteName.keyword_search:
            return [] if (focus or has_entity or has_filter) else ["query_focus"]
        if route is RouteName.structured_sql:
            return [] if structured_intent else ["filter_or_aggregate"]
        if route is RouteName.geo_search:
            if matches_any(lower, {"map", "geospatial", "spatial"}):
                return []
            if (
                "updated_location" in resolved_filters
                or _COORD_RE.search(lower)
                or _PROXIMITY_PLACE_RE.search(lower)
            ):
                return []
            if has_entity and matches_any(lower, _GEO_ANCHOR_WORDS):
                return []
            return ["location"]
        if route is RouteName.vector_search:
            return [] if (focus or has_entity or has_filter) else ["query_focus"]
        if route is RouteName.hybrid_search:
            # A broad query_focus is enough; do not demand a structured/geo signal.
            return [] if (focus or has_entity or has_filter) else ["query_focus"]
        if route is RouteName.disruption_analysis:
            if "company" in resolved_filters or "primary_oems" in resolved_filters or has_entity:
                return []
            return ["entity"]
        return []

    # -- clarification (no KB candidate options) ----------------------------
    def _clarification_route(
        self, raw_route, route, question, resolved_filters, search_filters, missing, source, actions
    ) -> FinalRoute:
        missing_fields = list(dict.fromkeys(missing))
        question_to_user = self._clarification_question(missing_fields)

        pending = PendingRouteState(
            original_question=question,
            proposed_route=route,
            raw_route=raw_route,
            known_fields={"entities": raw_route.entities, "query_focus": raw_route.query_focus},
            resolved_filters=resolved_filters,
            missing_fields=missing_fields,
            candidates={},  # never surface KB-derived candidate values (README §9)
            route_reason=raw_route.reason,
        )
        clarification = ClarificationRequest(
            needed=True,
            missing_fields=missing_fields,
            question_to_user=question_to_user,
            pending_route_state=pending,
        )
        needs_kb, needs_doc = route_needs_flags(RouteName.clarification_needed.value)
        return FinalRoute(
            question=question,
            route=RouteName.clarification_needed,
            confidence=raw_route.confidence,
            operation=Operation.ask_clarification.value,
            entities=raw_route.entities,
            raw_filters=raw_route.raw_filters,
            resolved_filters=resolved_filters,
            search_filters=search_filters,
            query_focus=raw_route.query_focus,
            needs_kb_access=needs_kb,
            needs_document_retrieval=needs_doc,
            retrieval_sources=route_retrieval_sources(needs_kb, needs_doc),
            missing_fields=missing_fields,
            validation_status="clarification_needed",
            route_source=source,
            clarification=clarification.model_dump(mode="json"),
            validation_actions=actions,
            reason=f"Missing required information: {', '.join(missing_fields)}.",
        )

    @staticmethod
    def _clarification_question(missing) -> str:
        """Generic, field-focused prompt. Never lists KB candidate values."""
        if missing:
            slot = missing[0]
            if slot in _MISSING_QUESTION:
                return _MISSING_QUESTION[slot]
            nice = slot.replace("_", " ")
            return (
                f"Which {nice} should this refer to: category, EV supply chain role, "
                "product/service, or industry group?"
            )
        return "Could you provide a bit more detail?"
