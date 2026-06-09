from georgia_ev_intelligence.route_execution.schemas import ExecutionResult, STATUS_SUCCESS
from georgia_ev_intelligence.streamlit_ui.services.route_chat_service import (
    RouteChatService,
    _context_center_for_geo,
    _context_entities_for_distance,
    _map_records_from_evidence,
    _provenance_rows_from_evidence,
    _sql_queries_from_evidence,
)


HISTORY = [
    ("user", "Map all Thermal Management suppliers in Georgia."),
    (
        "assistant",
        "Geocoded companies for the requested filters:\n"
        "1. freudenberg-nok - LaGrange, Troup County\n"
        "2. novelis inc. - Atlanta, Fulton County",
    ),
]
DISTANCE_HISTORY = [
    *HISTORY,
    ("user", "Distance of these companies to Kia georgia"),
    (
        "assistant",
        "Distances to Kia Georgia Inc.:\n"
        "1. Novelis Inc.: 77.3 miles - Atlanta, Fulton County\n"
        "2. Freudenberg-NOK: 6.7 miles - LaGrange, Troup County",
    ),
]
PLAIN_LIST_HISTORY = [
    (
        "assistant",
        "Based on the retrieved evidence:\n"
        "1. f&p georgia manufacturing\n"
        "2. hollingsworth & vose co.\n"
        "3. hyundai motor group\n"
        "4. immi\n",
    ),
]


def test_extracts_numbered_companies_for_distance_follow_up():
    assert _context_entities_for_distance(
        "Distance of these companies to Kia georgia",
        HISTORY,
    ) == ["freudenberg-nok", "novelis inc."]


def test_extracts_colon_formatted_companies_for_nearby_follow_up():
    assert _context_entities_for_distance(
        "Which of these companies are near to Kia Georgia?",
        DISTANCE_HISTORY,
    ) == ["Novelis Inc.", "Freudenberg-NOK"]


def test_extracts_plain_numbered_companies_for_nearby_follow_up():
    assert _context_entities_for_distance(
        "Which of these companies are near to Kia Georgia?",
        PLAIN_LIST_HISTORY,
    ) == [
        "f&p georgia manufacturing",
        "hollingsworth & vose co.",
        "hyundai motor group",
        "immi",
    ]


def test_extracts_colon_formatted_companies_for_nearest_follow_up():
    assert _context_entities_for_distance(
        "Which of these companies is nearest?",
        DISTANCE_HISTORY,
    ) == ["Novelis Inc.", "Freudenberg-NOK"]


def test_extracts_center_from_prior_distance_answer():
    assert _context_center_for_geo(DISTANCE_HISTORY) == "Kia Georgia Inc."


def test_route_chat_service_passes_context_entities_to_executor():
    captured = {}

    class FakeFinalRoute:
        def model_dump(self, mode="json"):
            return {
                "question": "Distance of these companies to Kia georgia",
                "route": "geo_search",
                "operation": "distance_search",
                "entities": ["Kia georgia"],
                "resolved_filters": {
                    "updated_location": {"operator": "CONTAINS", "value": "Kia georgia"},
                },
                "validation_actions": [],
            }

    class FakeRouteService:
        def route(self, query, history=None):
            return FakeFinalRoute()

    def execute(route, use_llm=False):
        captured.update(route)
        return ExecutionResult(
            route="geo_search",
            status=STATUS_SUCCESS,
            answer="Distances returned.",
            evidence={"type": "geo_results", "rows": []},
        )

    service = RouteChatService(
        route_service_factory=lambda: FakeRouteService(),
        execute_route_fn=execute,
        use_llm=False,
    )
    result = service.answer(
        "Distance of these companies to Kia georgia",
        history=HISTORY,
    )

    assert result.error == ""
    assert captured["context_entities"] == ["freudenberg-nok", "novelis inc."]
    assert captured["operation"] == "distance_search"
    assert "updated_location" not in captured["resolved_filters"]


def test_map_records_from_evidence_keeps_every_geocoded_row():
    evidence = {
        "type": "geo_results",
        "rows": [
            {
                "company": "Kia Georgia Inc.",
                "updated_location": "West Point",
                "product_service": "Vehicle assembly",
                "latitude": 32.8,
                "longitude": -85.1,
            },
            # string coordinates (as PostGIS numeric may serialise) are accepted
            {
                "company": "Novelis Inc.",
                "updated_location": "Atlanta",
                "latitude": "33.7",
                "longitude": "-84.4",
            },
        ],
    }

    records = _map_records_from_evidence(evidence)

    assert [r["company"] for r in records] == ["Kia Georgia Inc.", "Novelis Inc."]
    assert records[0]["address"] == "West Point"
    assert records[1]["latitude"] == 33.7 and records[1]["longitude"] == -84.4


def test_map_records_skips_rows_without_usable_coordinates():
    evidence = {
        "type": "geo_results",
        "rows": [
            {"company": "No Coords", "latitude": None, "longitude": None},
            {"company": "NaN Coords", "latitude": "nan", "longitude": "nan"},
            {"company": "Good", "latitude": 34.0, "longitude": -84.0},
        ],
    }

    records = _map_records_from_evidence(evidence)

    assert [r["company"] for r in records] == ["Good"]


def test_map_records_empty_for_non_geo_evidence():
    assert _map_records_from_evidence(None) == []
    assert _map_records_from_evidence({"type": "count", "count": 5}) == []


def test_answer_attaches_map_records_from_evidence():
    class FakeFinalRoute:
        def model_dump(self, mode="json"):
            return {
                "question": "Map Thermal Management suppliers in Georgia.",
                "route": "geo_search",
                "operation": "filtered_points",
                "entities": [],
                "resolved_filters": {},
                "validation_actions": [],
            }

    def execute(route, use_llm=False):
        return ExecutionResult(
            route="geo_search",
            status=STATUS_SUCCESS,
            answer="Geocoded companies for the requested filters.",
            evidence={
                "type": "geo_results",
                "rows": [
                    {"company": "A", "latitude": 33.0, "longitude": -84.0},
                    {"company": "B", "latitude": 34.0, "longitude": -85.0},
                ],
            },
        )

    result = RouteChatService(
        route_service_factory=lambda: type("S", (), {"route": lambda self, q, history=None: FakeFinalRoute()})(),
        execute_route_fn=execute,
        use_llm=False,
    ).answer("Map Thermal Management suppliers in Georgia.")

    assert [r["company"] for r in result.map_records] == ["A", "B"]


def test_provenance_rows_classifies_record_evidence():
    for etype in ("structured_rows", "geo_results"):
        kind, rows = _provenance_rows_from_evidence(
            {"type": etype, "rows": [{"company": "A"}, "bad", {"company": "B"}]}
        )
        assert kind == "records"
        assert [r["company"] for r in rows] == ["A", "B"]


def test_provenance_rows_classifies_groups_and_count():
    kind, rows = _provenance_rows_from_evidence(
        {"type": "group_counts", "groups": [{"category": "Tier 1", "count": 3}]}
    )
    assert kind == "groups" and rows == [{"category": "Tier 1", "count": 3}]

    kind, rows = _provenance_rows_from_evidence({"type": "count", "count": 9})
    assert kind == "count" and rows == []

    assert _provenance_rows_from_evidence({"type": "document_chunks"}) == ("", [])
    assert _provenance_rows_from_evidence(None) == ("", [])


def test_sql_queries_from_structured_and_geo_evidence():
    structured = _sql_queries_from_evidence(
        {"type": "structured_rows", "sql_display": "SELECT 1;"}
    )
    assert structured == [{"label": "SQL query", "sql": "SELECT 1;"}]

    geo = _sql_queries_from_evidence(
        {
            "type": "geo_results",
            "spatial_operation": "ST_DWithin_company",
            "sql_commands": [
                {"label": "nearby_by_company", "sql_display": "SELECT geo;"},
            ],
        }
    )
    assert geo == [{"label": "nearby_by_company", "sql": "SELECT geo;"}]
    assert _sql_queries_from_evidence({"type": "count"}) == []


def test_answer_attaches_provenance_from_evidence():
    class FakeFinalRoute:
        def model_dump(self, mode="json"):
            return {
                "question": "List Georgia battery companies and their tier.",
                "route": "structured_sql",
                "operation": "list_records",
                "entities": [],
                "resolved_filters": {},
                "validation_actions": [],
            }

    def execute(route, use_llm=False):
        return ExecutionResult(
            route="structured_sql",
            status=STATUS_SUCCESS,
            answer="Found 1 record.",
            evidence={
                "type": "structured_rows",
                "columns": ["company", "ev_supply_chain_role"],
                "rows": [{"company": "SK Battery", "ev_supply_chain_role": "Battery Cell"}],
                "sql_display": "SELECT company, ev_supply_chain_role FROM parent_chunks;",
            },
        )

    result = RouteChatService(
        route_service_factory=lambda: type("S", (), {"route": lambda self, q, history=None: FakeFinalRoute()})(),
        execute_route_fn=execute,
        use_llm=False,
    ).answer("List Georgia battery companies and their tier.")

    assert result.evidence_kind == "records"
    assert [r["company"] for r in result.evidence_rows] == ["SK Battery"]
    assert result.sql_queries == [
        {"label": "SQL query", "sql": "SELECT company, ev_supply_chain_role FROM parent_chunks;"}
    ]


def test_nearest_follow_up_overrides_clarification_and_limits_to_one():
    captured = {}

    class FakeFinalRoute:
        def model_dump(self, mode="json"):
            return {
                "question": "Which of these companies is nearest?",
                "route": "clarification_needed",
                "operation": "ask_clarification",
                "entities": [],
                "resolved_filters": {
                    "primary_oems": {"operator": "CONTAINS", "value": "Kia Georgia Inc."},
                },
                "validation_actions": [],
                "missing_fields": ["location"],
                "clarification": {"needed": True},
            }

    class FakeRouteService:
        def route(self, query, history=None):
            return FakeFinalRoute()

    def execute(route, use_llm=False):
        captured.update(route)
        return ExecutionResult(
            route="geo_search",
            status=STATUS_SUCCESS,
            answer="Nearest returned.",
            evidence={"type": "geo_results", "rows": []},
        )

    result = RouteChatService(
        route_service_factory=lambda: FakeRouteService(),
        execute_route_fn=execute,
        use_llm=False,
    ).answer("Which of these companies is nearest?", history=DISTANCE_HISTORY)

    assert result.error == ""
    assert captured["route"] == "geo_search"
    assert captured["operation"] == "distance_search"
    assert captured["entities"] == ["Kia Georgia Inc."]
    assert captured["limit"] == 1
    assert captured["validation_status"] == "valid"
