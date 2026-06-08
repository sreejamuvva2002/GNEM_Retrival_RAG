"""Unit tests for structured_sql query construction (build-only, no database)."""
from __future__ import annotations

from georgia_ev_intelligence.route_execution.executors.structured_sql import (
    build_query,
    execute_structured_sql,
    format_sql_for_display,
)


def test_list_records_default_limit_and_company_column():
    route = {
        "route": "structured_sql",
        "operation": "list_records",
        "resolved_filters": {
            "category": {"operator": "EQUALS", "value": "Tier 1/2"},
            # A real (sub-state) location still builds a predicate.
            "updated_location": {"operator": "CONTAINS", "value": "Fulton County"},
        },
        "requested_columns": ["ev_supply_chain_role", "product_service"],
    }
    sql, params, columns, mode = build_query(route)

    assert mode == "list"
    assert columns == ["company", "ev_supply_chain_role", "product_service"]
    assert sql.startswith(
        "SELECT company, ev_supply_chain_role, product_service FROM parent_chunks WHERE "
    )
    assert "category = %s" in sql
    assert "updated_location ILIKE %s" in sql
    assert sql.rstrip(";").endswith("LIMIT 100")
    assert params == ["Tier 1/2", "%Fulton County%"]


def test_state_filter_builds_real_predicate():
    route = {
        "route": "structured_sql",
        "operation": "list_records",
        "resolved_filters": {
            "ev_supply_chain_role": {"operator": "CONTAINS", "value": "Battery Cell"},
            "state": {"operator": "EQUALS", "value": "Georgia"},
        },
    }
    sql, params, _columns, _mode = build_query(route)
    assert "state = %s" in sql
    assert "ev_supply_chain_role ILIKE %s" in sql
    assert params == ["%Battery Cell%", "Georgia"]


def test_or_contains_location_keeps_all_values():
    route = {
        "route": "structured_sql",
        "operation": "list_records",
        "resolved_filters": {
            "updated_location": {
                "operator": "OR_CONTAINS",
                "value": ["Georgia", "Fulton County"],
            },
        },
    }
    sql, params, _columns, _mode = build_query(route)
    assert "(updated_location ILIKE %s OR updated_location ILIKE %s)" in sql
    assert params == ["%Georgia%", "%Fulton County%"]


def test_list_records_no_filters_uses_default_columns():
    sql, params, columns, mode = build_query({"route": "structured_sql"})
    assert mode == "list"
    assert "company" in columns
    assert "WHERE" not in sql
    assert params == []


def test_count_records():
    route = {
        "operation": "count_records",
        "resolved_filters": {"category": {"operator": "EQUALS", "value": "Tier 1/2"}},
    }
    sql, params, columns, mode = build_query(route)
    assert mode == "count"
    assert sql == "SELECT COUNT(*) AS count FROM parent_chunks WHERE category = %s;"
    assert columns == ["count"]
    assert params == ["Tier 1/2"]


def test_group_records():
    route = {
        "operation": "group_records",
        "group_by": ["category"],
        "resolved_filters": {},
    }
    sql, params, columns, mode = build_query(route)
    assert mode == "group"
    assert columns == ["category", "count"]
    assert sql == (
        "SELECT category, COUNT(*) AS count FROM parent_chunks "
        "GROUP BY category ORDER BY count DESC;"
    )
    assert params == []


def test_total_employment_by_county_builds_sum_aggregate():
    route = {
        "question": "Which county has the highest total employment among Tier 1 suppliers?",
        "operation": "aggregate_records",
        "resolved_filters": {
            "category": {"operator": "EQUALS", "value": "Tier 1"},
        },
        "group_by": ["county"],
        "sort_by": ["employment DESC"],
        "limit": 1,
    }

    sql, params, columns, mode = build_query(route)

    assert mode == "aggregate"
    assert columns == ["county", "total_employment"]
    assert "regexp_match(updated_location, '([^,]+ County)', 'i')" in sql
    assert "SUM(employment) AS total_employment" in sql
    assert "category = %s" in sql
    assert "GROUP BY NULLIF(btrim(" in sql
    assert "ORDER BY total_employment DESC LIMIT 1" in sql
    assert params == ["Tier 1"]


def test_empty_structured_query_returns_empty_rows_without_hybrid_fallback(monkeypatch):
    class Cursor:
        description = [("company",)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, _sql, _params):
            return None

        def fetchall(self):
            return []

    class Connection:
        def cursor(self):
            return Cursor()

        def close(self):
            return None

    monkeypatch.setattr(
        "georgia_ev_intelligence.route_execution.executors.structured_sql.get_connection",
        Connection,
    )

    result = execute_structured_sql({
        "operation": "list_records",
        "resolved_filters": {
            "category": {"operator": "EQUALS", "value": "Does Not Exist"},
        },
        "requested_columns": ["category"],
    })

    assert result.answer == "Found 0 matching records."
    assert result.evidence["type"] == "structured_rows"
    assert result.evidence["rows"] == []
    assert "fallback" not in result.evidence


def test_explicit_limit_and_sort():
    route = {
        "operation": "list_records",
        "requested_columns": ["category"],
        "sort_by": ["company"],
        "limit": 5,
    }
    sql, _params, _columns, _mode = build_query(route)
    assert "ORDER BY company ASC" in sql
    assert sql.rstrip(";").endswith("LIMIT 5")


def test_sort_by_parses_direction_safely():
    route = {
        "operation": "list_records",
        "requested_columns": ["employment"],
        "sort_by": ["employment DESC"],
    }
    sql, _params, _columns, _mode = build_query(route)
    assert "ORDER BY employment DESC" in sql


def test_sort_by_rejects_unknown_column():
    import pytest

    from georgia_ev_intelligence.route_execution.column_allowlist import UnknownColumnError

    with pytest.raises(UnknownColumnError):
        build_query({"operation": "list_records", "sort_by": ["ssn DESC"]})


def test_format_sql_for_display_renders_bound_values():
    sql = "SELECT * FROM parent_chunks WHERE state = %s AND category = ANY(%s);"
    rendered = format_sql_for_display(sql, ["Georgia", ["Tier 1", "Tier 2"]])

    assert rendered == (
        "SELECT * FROM parent_chunks WHERE state = 'Georgia' "
        "AND category = ANY(ARRAY['Tier 1', 'Tier 2']);"
    )


def test_format_sql_for_display_escapes_quotes():
    rendered = format_sql_for_display("SELECT * FROM t WHERE company = %s;", ["O'Reilly"])
    assert rendered == "SELECT * FROM t WHERE company = 'O''Reilly';"
