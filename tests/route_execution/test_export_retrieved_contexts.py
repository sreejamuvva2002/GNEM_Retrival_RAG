from __future__ import annotations

import json

import pandas as pd

from georgia_ev_intelligence.route_execution.export_retrieved_contexts import (
    normalize_state_filter,
    run,
)
from georgia_ev_intelligence.route_execution.schemas import ExecutionResult, STATUS_SUCCESS


def test_normalize_state_filter_upgrades_legacy_location_filter():
    route = {
        "route": "structured_sql",
        "raw_filters": [{"field_hint": "state", "raw_value": "Georgia"}],
        "resolved_filters": {
            "category": {"operator": "CONTAINS", "value": "Tier 1/2"},
            "updated_location": {"operator": "CONTAINS", "value": "Georgia"},
        },
    }

    normalized = normalize_state_filter(route)

    assert normalized["resolved_filters"]["state"] == {
        "operator": "EQUALS",
        "value": "Georgia",
    }
    assert "updated_location" not in normalized["resolved_filters"]
    assert "updated_location" in route["resolved_filters"]


def test_run_exports_sql_structured_data_and_document_contexts(tmp_path):
    input_path = tmp_path / "routes.jsonl"
    output_path = tmp_path / "retrieved.xlsx"
    records = [
        {
            "question": "List Georgia suppliers",
            "final_route": {
                "question": "List Georgia suppliers",
                "route": "structured_sql",
                "raw_filters": [{"field_hint": "state", "raw_value": "Georgia"}],
                "resolved_filters": {
                    "updated_location": {"operator": "CONTAINS", "value": "Georgia"},
                },
            },
        },
        {
            "question": "What do suppliers manufacture?",
            "final_route": {
                "question": "What do suppliers manufacture?",
                "route": "hybrid_search",
                "resolved_filters": {},
            },
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    def fake_executor(route, *, use_llm=False):
        assert use_llm is False
        if route["route"] == "structured_sql":
            assert route["resolved_filters"]["state"]["value"] == "Georgia"
            return ExecutionResult(
                route="structured_sql",
                status=STATUS_SUCCESS,
                answer="Found one.",
                evidence={
                    "type": "structured_rows",
                    "rows": [{"company": "Example Co", "state": "Georgia"}],
                    "columns": ["company", "state"],
                    "sql": "SELECT company, state FROM parent_chunks WHERE state = %s;",
                    "sql_params": ["Georgia"],
                    "sql_display": (
                        "SELECT company, state FROM parent_chunks WHERE state = 'Georgia';"
                    ),
                },
            )
        return ExecutionResult(
            route="hybrid_search",
            status=STATUS_SUCCESS,
            answer="Found context.",
            evidence={
                "type": "document_chunks",
                "chunks": [{
                    "chunk_id": "chunk-1",
                    "parent_record_id": "parent-1",
                    "chunk_type": "identity",
                }],
                "parents": [{
                    "parent_record_id": "parent-1",
                    "source_row_id": "row-1",
                    "text": "Example Co manufactures battery components.",
                }],
            },
        )

    successes, failures = run(input_path, output_path, executor=fake_executor)

    assert (successes, failures) == (2, 0)
    assert output_path.exists()

    workbook = pd.ExcelFile(output_path)
    assert set(workbook.sheet_names) == {
        "summary",
        "sql_commands",
        "structured_data",
        "retrieved_contexts",
        "final_routes",
    }
    sql = pd.read_excel(output_path, sheet_name="sql_commands")
    structured = pd.read_excel(output_path, sheet_name="structured_data")
    contexts = pd.read_excel(output_path, sheet_name="retrieved_contexts")
    assert sql.loc[0, "sql_display"].endswith("state = 'Georgia';")
    assert structured.loc[0, "company"] == "Example Co"
    assert contexts.loc[0, "context"] == "Example Co manufactures battery components."
