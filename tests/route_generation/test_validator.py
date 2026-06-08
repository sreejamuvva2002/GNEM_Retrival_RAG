"""Tests for the router validator."""
import pytest

from georgia_ev_intelligence.route_generation.metadata.provider import (
    ColumnMetaView,
    IndexBackedProvider,
)
from georgia_ev_intelligence.route_generation.routing.validator import RouteValidator
from georgia_ev_intelligence.route_generation.schemas import (
    FinalRoute,
    RawFilter,
    RawRoute,
    RouteName,
    SearchFilter,
)
from georgia_ev_intelligence.route_generation.utils.text_normalization import (
    normalize_question,
)


def _validate(provider, raw_route, question, **kw):
    return RouteValidator(provider).validate(raw_route, normalize_question(question), **kw)


class TestHappyPaths:
    def test_structured_sql_with_resolved_filter(self, fixture_metadata):
        raw = RawRoute(
            route=RouteName.structured_sql, confidence=0.9, operation="list",
            raw_filters=[RawFilter(field_hint="tier", raw_value="Tier 2/3")],
            reason="filter by tier",
        )
        final = _validate(fixture_metadata, raw, "list Tier 2/3 suppliers")
        assert final.validation_status == "valid"
        assert final.route == RouteName.structured_sql
        # Tier/category values are preserved verbatim and matched exactly.
        assert final.resolved_filters["category"] == {"operator": "EQUALS", "value": "Tier 2/3"}
        assert final.needs_kb_access is True
        assert final.needs_document_retrieval is False

    def test_vector_search_needs_flags(self, fixture_metadata):
        raw = RawRoute(route=RouteName.vector_search, confidence=0.8,
                       query_focus="battery suppliers", reason="semantic")
        final = _validate(fixture_metadata, raw, "tell me about battery suppliers")
        assert final.validation_status == "valid"
        assert (final.needs_kb_access, final.needs_document_retrieval) == (True, True)

    def test_no_retrieval_never_clarifies_even_low_confidence(self, fixture_metadata):
        raw = RawRoute(route=RouteName.no_retrieval, confidence=0.3, reason="greeting")
        final = _validate(fixture_metadata, raw, "hello")
        assert final.validation_status == "valid"
        assert final.needs_kb_access is False


class TestRouteCorrection:
    def test_aggregate_signal_upgrades_vector_to_structured(self, fixture_metadata):
        # LLM wrongly chose vector_search for a counting question.
        raw = RawRoute(route=RouteName.vector_search, confidence=0.5, reason="wrong")
        final = _validate(fixture_metadata, raw, "how many companies are there")
        assert final.route == RouteName.structured_sql
        assert final.route_source == "validator_corrected"
        assert final.validation_status == "valid"  # aggregate intent satisfies structured_sql

    def test_disruption_signal_overrides(self, fixture_metadata):
        raw = RawRoute(route=RouteName.vector_search, confidence=0.6,
                       entities=["SK Battery America"], reason="x")
        final = _validate(fixture_metadata, raw, "what are alternatives to SK Battery America")
        assert final.route == RouteName.disruption_analysis
        assert final.route_source == "validator_corrected"

    def test_tier_value_reverse_rescued_to_category(self, fixture_metadata):
        # The reported bug: the LLM drops a tier classification on
        # ev_supply_chain_role, so execution searches the wrong column -> 0 rows.
        # The validator reverse-rescues it onto category (KB-free, by wording) and
        # mirrors it as an audit search_filter with category-first candidates.
        raw = RawRoute(
            route=RouteName.exact_lookup,
            confidence=0.9,
            entities=["suppliers"],
            raw_filters=[
                RawFilter(field_hint="ev_supply_chain_role", raw_value="Tier 1/2"),
                RawFilter(field_hint="location", raw_value="Georgia"),
            ],
            reason="bad LLM field mapping",
        )

        final = _validate(
            fixture_metadata,
            raw,
            'Show all "Tier 1/2" suppliers in Georgia.',
        )

        # "show all" is multi-entity -> structured_sql, never exact_lookup.
        assert final.route == RouteName.structured_sql
        assert final.route_source == "validator_corrected"
        assert final.validation_status == "valid"
        # Value moved off ev_supply_chain_role and onto category, preserved verbatim.
        assert "ev_supply_chain_role" not in final.resolved_filters
        assert final.resolved_filters["category"] == {
            "operator": "EQUALS",
            "value": "Tier 1/2",
        }
        assert final.resolved_filters["updated_location"]["operator"] == "CONTAINS"
        # Audit search_filter with category tried first.
        tier_sf = [sf for sf in final.search_filters if sf.raw_value == "Tier 1/2"]
        assert tier_sf and tier_sf[0].field_candidates[0] == "category"
        # The move is recorded for audit.
        assert any("to category" in a for a in final.validation_actions)


class TestGeoRouting:
    """Geo intent stays on the PostGIS execution route."""

    def test_map_in_location_stays_geo_search(self, fixture_metadata):
        raw = RawRoute(
            route=RouteName.geo_search, confidence=0.9,
            raw_filters=[
                RawFilter(field_hint="role", raw_value="Thermal Management"),
                RawFilter(field_hint="location", raw_value="Georgia"),
            ],
            reason="map",
        )
        final = _validate(
            fixture_metadata, raw,
            "Map all Thermal Management suppliers in Georgia and show which Primary OEMs.",
        )
        assert final.route == RouteName.geo_search
        assert final.validation_status == "valid"

    def test_map_wording_upgrades_vector_to_geo_search(self, fixture_metadata):
        raw = RawRoute(
            route=RouteName.vector_search,
            confidence=0.9,
            raw_filters=[RawFilter(field_hint="location", raw_value="Georgia")],
            reason="wrong route",
        )
        final = _validate(fixture_metadata, raw, "Map companies in Georgia.")
        assert final.route == RouteName.geo_search
        assert final.route_source == "validator_corrected"
        assert any("geospatial wording" in action for action in final.validation_actions)

    def test_county_list_upgrades_vector_to_geo_search(self, fixture_metadata):
        raw = RawRoute(
            route=RouteName.vector_search,
            confidence=0.9,
            raw_filters=[RawFilter(field_hint="location", raw_value="Troup County")],
            reason="wrong route",
        )
        final = _validate(fixture_metadata, raw, "List companies in Troup County.")
        assert final.route == RouteName.geo_search

    def test_county_aggregate_remains_structured_sql(self, fixture_metadata):
        raw = RawRoute(route=RouteName.geo_search, confidence=0.9, reason="wrong route")
        final = _validate(
            fixture_metadata,
            raw,
            "Which county has the highest total employment?",
        )
        assert final.route == RouteName.structured_sql

    def test_genuine_proximity_stays_geo_search(self, fixture_metadata):
        raw = RawRoute(
            route=RouteName.geo_search, confidence=0.9,
            raw_filters=[RawFilter(field_hint="location", raw_value="West Point")],
            reason="geo",
        )
        final = _validate(fixture_metadata, raw, "companies near West Point")
        assert final.route == RouteName.geo_search


class TestFieldRejection:
    def test_non_filterable_field_dropped(self, fixture_metadata):
        raw = RawRoute(
            route=RouteName.structured_sql, confidence=0.9, operation="list",
            raw_filters=[RawFilter(field_hint="classification_method", raw_value="llm")],
            reason="x",
        )
        final = _validate(fixture_metadata, raw, "list records")
        # classification_method is not filterable -> dropped from filters.
        assert "classification_method" not in final.resolved_filters
        # "list" is a valid structured intent (list all) -> no clarification.
        assert final.validation_status == "valid"
        assert final.route == RouteName.structured_sql

    def test_truly_empty_structured_request_clarifies(self, fixture_metadata):
        # No filter, no list/aggregate/sort/column intent -> genuinely missing.
        raw = RawRoute(
            route=RouteName.structured_sql, confidence=0.9, operation="list",
            raw_filters=[RawFilter(field_hint="classification_method", raw_value="llm")],
            reason="x",
        )
        final = _validate(fixture_metadata, raw, "tell me about these records")
        assert final.validation_status == "clarification_needed"
        assert "filter_or_aggregate" in final.missing_fields


class TestClarification:
    def test_unmatched_value_is_preserved_not_clarified(self, fixture_metadata):
        # "Tier 1/Tier 2" does not match a stored category. Old behavior asked for
        # clarification with KB options; new behavior preserves it as an OR search.
        raw = RawRoute(
            route=RouteName.structured_sql, confidence=0.9,
            raw_filters=[RawFilter(field_hint="tier", raw_value="Tier 1/Tier 2")],
            reason="x",
        )
        final = _validate(fixture_metadata, raw, "list Tier 1 or Tier 2 records")
        assert final.validation_status == "valid"
        assert final.resolved_filters["category"] == {
            "operator": "OR_EQUALS",
            "value": ["Tier 1", "Tier 2"],
        }

    def test_clarification_never_lists_kb_candidate_options(self, fixture_metadata):
        # Even when we DO clarify (missing location), no KB values are surfaced.
        raw = RawRoute(route=RouteName.geo_search, confidence=0.9, reason="geo")
        final = _validate(fixture_metadata, raw, "find nearby companies")
        clar = final.clarification
        assert clar["pending_route_state"]["candidates"] == {}
        question = clar["question_to_user"].lower()
        for kb_value in ("oem", "tier 1", "tier 1/2", "tier 2/3", "footprint"):
            assert kb_value not in question

    def test_missing_location_clarifies(self, fixture_metadata):
        raw = RawRoute(route=RouteName.geo_search, confidence=0.9,
                       operation="nearby", reason="geo")
        final = _validate(fixture_metadata, raw, "find nearby companies")
        assert final.validation_status == "clarification_needed"
        assert "location" in final.missing_fields

    def test_low_confidence_clarifies(self, fixture_metadata):
        raw = RawRoute(route=RouteName.vector_search, confidence=0.2,
                       query_focus="batteries", reason="unsure")
        final = _validate(fixture_metadata, raw, "tell me about batteries")
        assert final.validation_status == "clarification_needed"

    def test_pending_state_carries_raw_route(self, fixture_metadata):
        raw = RawRoute(route=RouteName.geo_search, confidence=0.9, reason="geo")
        final = _validate(fixture_metadata, raw, "find nearby companies")
        pending = final.clarification["pending_route_state"]
        assert pending["original_question"] == "find nearby companies"
        assert pending["raw_route"]["route"] == "geo_search"  # enum serialized to string


class TestOrFilters:
    """Repeated raw_filters for one field become an OR group, never overwrite."""

    def test_repeated_text_field_becomes_or_contains(self, fixture_metadata):
        raw = RawRoute(
            route=RouteName.structured_sql, confidence=0.9,
            raw_filters=[
                RawFilter(field_hint="role", raw_value="Battery Cell"),
                RawFilter(field_hint="role", raw_value="Battery Pack"),
            ],
            reason="x",
        )
        final = _validate(fixture_metadata, raw, "Which Battery Cell or Battery Pack suppliers?")
        assert final.validation_status == "valid"
        # The last value did NOT overwrite the first — both are preserved.
        assert final.resolved_filters["ev_supply_chain_role"] == {
            "operator": "OR_CONTAINS",
            "value": ["Battery Cell", "Battery Pack"],
        }

    def test_repeated_category_values_become_or_equals(self, fixture_metadata):
        # KB-free: repeated category values OR-merge as exact alternatives.
        raw = RawRoute(
            route=RouteName.structured_sql, confidence=0.9,
            raw_filters=[
                RawFilter(field_hint="tier", raw_value="Tier 1"),
                RawFilter(field_hint="tier", raw_value="Tier 2/3"),
            ],
            reason="x",
        )
        final = _validate(fixture_metadata, raw, "list Tier 1 and Tier 2/3 records")
        assert final.resolved_filters["category"] == {
            "operator": "OR_EQUALS",
            "value": ["Tier 1", "Tier 2/3"],
        }


class TestValuePreservation:
    """Unmatched values are preserved without any KB / metadata lookup."""

    @staticmethod
    def _no_value_provider():
        # Exact field with NO unique_values: resolution cannot consult the KB.
        return IndexBackedProvider({
            "category": ColumnMetaView(
                field="category", match_type="exact", is_filterable=True, unique_values=[],
            ),
            "updated_location": ColumnMetaView(
                field="updated_location", match_type="partial", is_filterable=True,
                unique_values=[],
            ),
        })

    def test_tier_value_preserved_without_metadata_lookup(self):
        provider = self._no_value_provider()
        raw = RawRoute(
            route=RouteName.structured_sql, confidence=0.9,
            raw_filters=[RawFilter(field_hint="tier", raw_value="Tier 1/2")],
            reason="x",
        )
        final = _validate(provider, raw, "list Tier 1/2 suppliers")
        assert final.validation_status == "valid"
        # No clarification, no candidates — the slash value is kept verbatim.
        assert final.resolved_filters["category"] == {
            "operator": "EQUALS",
            "value": "Tier 1/2",
        }


class TestRequestedColumns:
    """A field asked as OUTPUT must not be treated as a missing/junk filter."""

    def test_output_column_not_treated_as_filter(self, fixture_metadata):
        # q007: "...what is its EV Supply Chain Role?" — the LLM echoes the field
        # name as a filter value; it must become a requested column, not a filter.
        raw = RawRoute(
            route=RouteName.structured_sql, confidence=0.9,
            raw_filters=[
                RawFilter(field_hint="location", raw_value="Gwinnett County"),
                RawFilter(field_hint="ev_supply_chain_role", raw_value="EV Supply Chain Role"),
            ],
            reason="x",
        )
        final = _validate(
            fixture_metadata, raw,
            "In Gwinnett County, which company has the highest Employment "
            "and what is its EV Supply Chain Role?",
        )
        assert final.validation_status == "valid"
        assert "ev_supply_chain_role" in final.requested_columns
        assert "ev_supply_chain_role" not in final.resolved_filters
        assert final.resolved_filters["updated_location"]["operator"] == "CONTAINS"
        # "highest Employment" -> ranking
        assert final.sort_by == ["employment DESC"]
        assert final.limit == 1


class TestEmploymentNumeric:
    """Employment comparisons become numeric operators, not CONTAINS."""

    @pytest.mark.parametrize(
        "question, expected",
        [
            ("Georgia suppliers with employment over 300", {"operator": "GT", "value": 300}),
            ("companies with more than 300 employees", {"operator": "GT", "value": 300}),
            ("companies with fewer than 200 employees", {"operator": "LT", "value": 200}),
            ("companies with under 200 employees", {"operator": "LT", "value": 200}),
            ("suppliers with at least 1,000 employees", {"operator": "GTE", "value": 1000}),
        ],
    )
    def test_numeric_operator(self, fixture_metadata, question, expected):
        raw = RawRoute(route=RouteName.structured_sql, confidence=0.9, reason="x")
        final = _validate(fixture_metadata, raw, question)
        assert final.resolved_filters["employment"] == expected

    def test_top_n_by_employment_ranking(self, fixture_metadata):
        raw = RawRoute(route=RouteName.structured_sql, confidence=0.9, reason="x")
        final = _validate(fixture_metadata, raw, "Top 10 Georgia companies by employment size")
        assert final.sort_by == ["employment DESC"]
        assert final.limit == 10

    def test_highest_total_employment_by_county_becomes_grouped_aggregate(
        self, fixture_metadata
    ):
        raw = RawRoute(
            route=RouteName.structured_sql,
            confidence=1.0,
            operation="list_records",
            raw_filters=[
                RawFilter(field_hint="category", raw_value="Tier 1"),
                RawFilter(field_hint="ev_supply_chain_role", raw_value="supplier"),
            ],
            requested_columns=["updated_location"],
            group_by=["county"],
            sort_by=["employment DESC"],
            limit=1,
            reason="highest county employment",
        )

        final = _validate(
            fixture_metadata,
            raw,
            "Which county have the highest total Employment among Tier 1 suppliers only?",
        )

        assert final.operation == "aggregate_records"
        assert final.group_by == ["county"]
        assert final.sort_by == ["employment DESC"]
        assert final.limit == 1
        assert final.resolved_filters == {
            "category": {"operator": "EQUALS", "value": "Tier 1"},
        }
        assert any("dropped generic" in action for action in final.validation_actions)
        assert any("aggregate_records" in action for action in final.validation_actions)


class TestExactLookupCorrection:
    """Multi-company questions must never be classified as exact_lookup."""

    @pytest.mark.parametrize(
        "question",
        [
            "Which companies are classified as Direct Manufacturer?",
            "List all Georgia battery suppliers",
            "Identify all Vehicle Assembly facilities in Georgia",
            "Find Georgia companies that make wiring harnesses",
        ],
    )
    def test_multi_company_not_exact_lookup(self, fixture_metadata, question):
        raw = RawRoute(
            route=RouteName.exact_lookup, confidence=0.9, entities=["companies"], reason="x",
        )
        final = _validate(fixture_metadata, raw, question)
        assert final.route != RouteName.exact_lookup
        assert final.route == RouteName.structured_sql
        assert final.route_source == "validator_corrected"


class TestCrossFieldRescue:
    """A value on a likely-wrong field is rescued, not clarified."""

    def test_thermal_management_rescued_off_category(self, fixture_metadata):
        raw = RawRoute(
            route=RouteName.structured_sql, confidence=0.9,
            raw_filters=[RawFilter(field_hint="category", raw_value="Thermal Management")],
            reason="x",
        )
        final = _validate(
            fixture_metadata, raw,
            "Map all Thermal Management suppliers in Georgia and show which Primary OEMs.",
        )
        assert final.validation_status == "valid"
        assert "category" not in final.resolved_filters
        # Moved to a searchable text field, preserved behind CONTAINS.
        rescued = {"ev_supply_chain_role", "product_service"} & set(final.resolved_filters)
        assert rescued
        field = rescued.pop()
        assert final.resolved_filters[field] == {
            "operator": "CONTAINS",
            "value": "Thermal Management",
        }


class TestSchemaListCoercion:
    """sort_by=None (and friends) must coerce to [] instead of raising."""

    def test_raw_route_sort_by_none_coerces(self):
        raw = RawRoute.model_validate({
            "route": "structured_sql", "confidence": 0.9,
            "sort_by": None, "group_by": None, "requested_columns": None,
            "entities": None, "raw_filters": None, "missing_fields": None,
        })
        assert raw.sort_by == []
        assert raw.group_by == []
        assert raw.requested_columns == []
        assert raw.entities == []
        assert raw.raw_filters == []

    def test_raw_route_scalar_string_wraps_to_list(self):
        raw = RawRoute.model_validate({
            "route": "structured_sql", "confidence": 0.9, "sort_by": "employment DESC",
        })
        assert raw.sort_by == ["employment DESC"]

    def test_final_route_sort_by_none_coerces(self):
        final = FinalRoute.model_validate({
            "question": "q", "route": "structured_sql", "confidence": 0.9,
            "sort_by": None, "group_by": None, "requested_columns": None,
            "needs_kb_access": True, "needs_document_retrieval": False,
            "validation_status": "valid", "route_source": "validator_corrected",
        })
        assert final.sort_by == []
        assert final.group_by == []

    def test_final_route_new_list_fields_none_coerce(self):
        final = FinalRoute.model_validate({
            "question": "q", "route": "structured_sql", "confidence": 0.9,
            "search_filters": None, "retrieval_sources": None, "validation_actions": None,
            "needs_kb_access": True, "needs_document_retrieval": False,
            "validation_status": "valid", "route_source": "validator_corrected",
        })
        assert final.search_filters == []
        assert final.retrieval_sources == []
        assert final.validation_actions == []


class TestSearchFilterModel:
    def test_defaults(self):
        sf = SearchFilter(raw_value="Tier 1/2")
        assert sf.field_candidates == []
        assert sf.operator == "CONTAINS"
        assert sf.fallback_policy == "try_in_order_then_multi_field"
        assert sf.source_text is None


class TestSearchFilterFallback:
    """Uncertain / unmappable filters become search_filters, never silently dropped."""

    def test_unknown_field_hint_becomes_search_filter(self, fixture_metadata):
        # A real value with a hint that maps to no KB column -> preserved as a
        # search_filter with candidate column NAMES (never KB values).
        raw = RawRoute(
            route=RouteName.structured_sql, confidence=0.9,
            raw_filters=[RawFilter(field_hint="some_made_up_field", raw_value="Thermal Management")],
            reason="x",
        )
        final = _validate(fixture_metadata, raw, "list Thermal Management companies")
        assert final.validation_status == "valid"
        assert any(sf.raw_value == "Thermal Management" for sf in final.search_filters)
        assert any("created search_filter" in a for a in final.validation_actions)

    def test_non_filterable_mapped_hint_dropped_without_search_filter(self, fixture_metadata):
        # classification_method is a real but non-filterable column -> dropped, and
        # NOT turned into a search_filter (it is a known, deliberately excluded field).
        raw = RawRoute(
            route=RouteName.structured_sql, confidence=0.9, operation="list",
            raw_filters=[RawFilter(field_hint="classification_method", raw_value="llm")],
            reason="x",
        )
        final = _validate(fixture_metadata, raw, "list records")
        assert "classification_method" not in final.resolved_filters
        assert final.search_filters == []


class TestValidationActions:
    def test_route_correction_recorded(self, fixture_metadata):
        raw = RawRoute(route=RouteName.vector_search, confidence=0.5, reason="wrong")
        final = _validate(fixture_metadata, raw, "how many companies are there")
        assert final.route == RouteName.structured_sql
        assert any("vector_search -> structured_sql" in a for a in final.validation_actions)


class TestHybridEvidence:
    def test_structured_filter_plus_web_evidence_routes_hybrid(self, fixture_metadata):
        raw = RawRoute(
            route=RouteName.structured_sql, confidence=0.9,
            raw_filters=[RawFilter(field_hint="tier", raw_value="Tier 1")],
            reason="x",
        )
        final = _validate(
            fixture_metadata, raw,
            "List Tier 1 suppliers and what web evidence supports their OEM links.",
        )
        assert final.route == RouteName.hybrid_search
        assert final.retrieval_sources == ["structured_db", "document_chunks"]
        assert any("hybrid_search" in a for a in final.validation_actions)


class TestRetrievalSources:
    def test_structured_sql_sources(self, fixture_metadata):
        raw = RawRoute(
            route=RouteName.structured_sql, confidence=0.9,
            raw_filters=[RawFilter(field_hint="tier", raw_value="Tier 1")],
            reason="x",
        )
        final = _validate(fixture_metadata, raw, "list Tier 1 suppliers")
        assert final.retrieval_sources == ["structured_db"]

    def test_vector_search_sources(self, fixture_metadata):
        # vector_search reads both structured KB and document chunks (needs flags).
        raw = RawRoute(route=RouteName.vector_search, confidence=0.9,
                       query_focus="battery suppliers", reason="semantic")
        final = _validate(fixture_metadata, raw, "tell me about battery suppliers")
        assert final.retrieval_sources == ["structured_db", "document_chunks"]

    def test_no_retrieval_sources_empty(self, fixture_metadata):
        raw = RawRoute(route=RouteName.no_retrieval, confidence=0.9, reason="greeting")
        final = _validate(fixture_metadata, raw, "hello")
        assert final.retrieval_sources == []
