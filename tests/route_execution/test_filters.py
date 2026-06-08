"""Unit tests for the parameterized filter builder (no database)."""
from __future__ import annotations

import pytest

from georgia_ev_intelligence.route_execution.column_allowlist import UnknownColumnError
from georgia_ev_intelligence.route_execution.filters import (
    FilterError,
    build_predicate,
    build_where,
)


def test_equals():
    sql, params = build_predicate("category", {"operator": "EQUALS", "value": "Tier 1/2"})
    assert sql == "category = %s"
    assert params == ["Tier 1/2"]


def test_contains_wraps_value():
    sql, params = build_predicate("updated_location", {"operator": "CONTAINS", "value": "Georgia"})
    assert sql == "updated_location ILIKE %s"
    assert params == ["%Georgia%"]


def test_or_contains_expands_each_value():
    sql, params = build_predicate(
        "ev_supply_chain_role",
        {"operator": "OR_CONTAINS", "value": ["Battery Cell", "Battery Pack"]},
    )
    assert sql == "(ev_supply_chain_role ILIKE %s OR ev_supply_chain_role ILIKE %s)"
    assert params == ["%Battery Cell%", "%Battery Pack%"]


def test_in_uses_any_array():
    sql, params = build_predicate("category", {"operator": "IN", "value": ["a", "b"]})
    assert sql == "category = ANY(%s)"
    assert params == [["a", "b"]]


@pytest.mark.parametrize(
    "operator,sql_op",
    [("GT", ">"), ("LT", "<"), ("GTE", ">="), ("LTE", "<=")],
)
def test_numeric_comparisons(operator, sql_op):
    sql, params = build_predicate("employment", {"operator": operator, "value": 300})
    assert sql == f"employment::numeric {sql_op} %s"
    assert params == [300]


def test_between_requires_two_bounds():
    sql, params = build_predicate("employment", {"operator": "BETWEEN", "value": [100, 500]})
    assert sql == "employment::numeric BETWEEN %s AND %s"
    assert params == [100, 500]

    with pytest.raises(FilterError):
        build_predicate("employment", {"operator": "BETWEEN", "value": [100]})


def test_clarification_needed_rejected():
    with pytest.raises(FilterError):
        build_predicate("category", {"operator": "CLARIFICATION_NEEDED", "value": None})


def test_unknown_column_rejected():
    with pytest.raises(UnknownColumnError):
        build_predicate("ssn", {"operator": "EQUALS", "value": "x"})


def test_unsupported_operator_rejected():
    with pytest.raises(FilterError):
        build_predicate("category", {"operator": "REGEX", "value": "x"})


def test_substring_on_numeric_column_rejected():
    # Router occasionally emits e.g. employment CONTAINS "highest" — not executable.
    with pytest.raises(FilterError):
        build_predicate("employment", {"operator": "CONTAINS", "value": "highest"})


def test_build_where_ands_predicates():
    where_sql, params = build_where({
        "category": {"operator": "EQUALS", "value": "Tier 1/2"},
        "updated_location": {"operator": "CONTAINS", "value": "Georgia"},
    })
    assert where_sql == "category = %s AND updated_location ILIKE %s"
    assert params == ["Tier 1/2", "%Georgia%"]


def test_build_where_empty():
    where_sql, params = build_where({})
    assert where_sql == ""
    assert params == []
