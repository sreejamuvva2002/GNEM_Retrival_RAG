from __future__ import annotations

from georgia_ev_intelligence.route_execution import answer_formatter
from georgia_ev_intelligence.route_execution import executor as route_executor
from georgia_ev_intelligence.route_execution.schemas import ExecutionResult, STATUS_SUCCESS


def test_grounded_evidence_keeps_rows_and_excludes_sql_debug_fields():
    evidence = {
        "type": "structured_rows",
        "columns": ["company", "category"],
        "rows": [
            {"company": "Battery Co", "category": "Tier 1"},
            {"company": "Pack Co", "category": "Tier 2"},
        ],
        "sql": "SELECT company, category FROM parent_chunks;",
        "sql_params": ["Georgia"],
        "sql_display": "SELECT ...",
        "sql_commands": [{"sql": "SELECT ..."}],
    }

    grounded = answer_formatter._grounded_evidence(evidence)

    assert grounded == {
        "type": "structured_rows",
        "columns": ["company", "category"],
        "rows": [
            {"company": "Battery Co", "category": "Tier 1"},
            {"company": "Pack Co", "category": "Tier 2"},
        ],
    }
    assert "sql" not in grounded


def test_grounded_evidence_strips_map_only_coordinates_from_rows():
    """lat/lon are fetched for the map but must not reach the answer LLM."""
    grounded = answer_formatter._grounded_evidence({
        "type": "structured_rows",
        "columns": ["company", "ev_supply_chain_role"],
        "rows": [
            {
                "company": "SK Battery",
                "ev_supply_chain_role": "Battery Cell",
                "latitude": 33.5,
                "longitude": -82.1,
            },
        ],
    })

    assert grounded["rows"] == [
        {"company": "SK Battery", "ev_supply_chain_role": "Battery Cell"}
    ]


def test_grounded_evidence_keeps_full_document_parent_contexts():
    long_context = "battery context " * 100
    grounded = answer_formatter._grounded_evidence({
        "type": "document_chunks",
        "chunks": [{"chunk_id": "c1", "parent_record_id": "p1"}],
        "parents": [{"parent_record_id": "p1", "text": long_context}],
        "structured_sql": "SELECT ...",
    })

    assert grounded["parents"][0]["text"] == long_context
    assert "structured_sql" not in grounded


def test_grounded_evidence_keeps_grouped_aggregate_results():
    grounded = answer_formatter._grounded_evidence({
        "type": "group_aggregates",
        "groups": [{"county": "Troup County", "total_employment": 2435}],
        "group_by": ["county"],
        "aggregate_column": "total_employment",
        "sql": "SELECT ...",
    })

    assert grounded == {
        "type": "group_aggregates",
        "groups": [{"county": "Troup County", "total_employment": 2435}],
        "group_by": ["county"],
        "aggregate_column": "total_employment",
    }


def test_build_prompt_contains_filters_rows_and_requested_columns_without_sql():
    prompt = answer_formatter._build_prompt(
        "Which Georgia battery companies are Tier 1?",
        route_context={
            "resolved_filters": {
                "state": {"operator": "EQUALS", "value": "Georgia"},
            },
            "requested_columns": ["category"],
        },
        grounded_evidence={
            "type": "structured_rows",
            "rows": [{"company": "Battery Co", "category": "Tier 1"}],
        },
        deterministic_answer="Found 1 matching record.",
    )

    assert '"state"' in prompt
    assert '"requested_columns"' in prompt
    assert '"Battery Co"' in prompt
    assert "Found 1 matching record." in prompt
    assert "SELECT" not in prompt


def test_build_prompt_forbids_reinterpreting_returned_rows():
    prompt = answer_formatter._build_prompt(
        "Map Thermal Management suppliers.",
        route_context={"route": "geo_search"},
        grounded_evidence={"type": "geo_results", "rows": [{"company": "Example"}]},
        deterministic_answer="Example",
    )

    assert "every returned row already satisfies the validated filters" in prompt
    assert "never exclude or re-filter a row" in prompt


def test_execute_route_passes_validated_route_and_evidence_to_answer_llm(monkeypatch):
    route = {
        "question": "Which Georgia companies are Tier 1?",
        "route": "structured_sql",
        "resolved_filters": {
            "state": {"operator": "EQUALS", "value": "Georgia"},
        },
        "requested_columns": ["category"],
    }
    evidence = {
        "type": "structured_rows",
        "columns": ["company", "category"],
        "rows": [{"company": "Battery Co", "category": "Tier 1"}],
    }
    captured = {}

    monkeypatch.setitem(
        route_executor._DISPATCH,
        "structured_sql",
        lambda _route: ExecutionResult(
            route="structured_sql",
            status=STATUS_SUCCESS,
            answer="Found 1 matching record.",
            evidence=evidence,
        ),
    )

    def fake_llm_answer(question, actual_evidence, deterministic, *, final_route=None):
        captured.update({
            "question": question,
            "evidence": actual_evidence,
            "deterministic": deterministic,
            "final_route": final_route,
        })
        return "Grounded final answer."

    monkeypatch.setattr(route_executor.fmt, "llm_answer", fake_llm_answer)

    result = route_executor.execute_route(route, use_llm=True)

    assert result.answer == "Grounded final answer."
    assert captured["question"] == route["question"]
    assert captured["evidence"] == evidence
    assert captured["final_route"] == route
