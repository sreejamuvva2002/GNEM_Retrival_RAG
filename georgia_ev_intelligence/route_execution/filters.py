"""Translate a route's ``resolved_filters`` into a safe parameterized WHERE clause.

``resolved_filters`` maps a real column name to ``{"operator": <FilterOperator>,
"value": <canonical value or list>}`` (see route_generation/schemas.py). This
module converts that mapping into ``(where_sql, params)`` where every column is
allowlist-checked and every value is a bound parameter — the executor never
builds SQL from raw values.
"""
from __future__ import annotations

from typing import Any

from .column_allowlist import NUMERIC_COLUMNS, ensure_allowed

# Substring (ILIKE) operators are meaningless on NUMERIC columns.
_SUBSTRING_OPERATORS = {"CONTAINS", "OR_CONTAINS"}


class FilterError(ValueError):
    """Raised when a filter cannot be turned into a safe SQL clause."""


def _ilike_term(value: Any) -> str:
    """Wrap a value for a case-insensitive substring (ILIKE) match."""
    return f"%{value}%"


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def build_predicate(column: str, spec: dict[str, Any]) -> tuple[str, list[Any]]:
    """Build a single ``(sql, params)`` predicate for one resolved filter.

    ``column`` is validated against the allowlist; ``spec`` carries ``operator``
    and ``value``.
    """
    ensure_allowed(column)

    operator = str(spec.get("operator", "")).upper()
    value = spec.get("value")

    if operator in _SUBSTRING_OPERATORS and column in NUMERIC_COLUMNS:
        raise FilterError(
            f"Cannot apply substring match ({operator}) to numeric column "
            f"{column!r} (value={value!r}); this route's filter is not executable."
        )

    if operator == "EQUALS":
        return f"{column} = %s", [value]

    if operator in {"IN", "OR_EQUALS"}:
        values = _as_list(value)
        return f"{column} = ANY(%s)", [values]

    if operator == "CONTAINS":
        return f"{column} ILIKE %s", [_ilike_term(value)]

    if operator == "OR_CONTAINS":
        values = _as_list(value)
        if not values:
            raise FilterError(f"OR_CONTAINS on {column!r} has no values")
        clause = " OR ".join(f"{column} ILIKE %s" for _ in values)
        return f"({clause})", [_ilike_term(v) for v in values]

    if operator in {"GT", "LT", "GTE", "LTE"}:
        sql_op = {"GT": ">", "LT": "<", "GTE": ">=", "LTE": "<="}[operator]
        return f"{column}::numeric {sql_op} %s", [value]

    if operator == "BETWEEN":
        bounds = _as_list(value)
        if len(bounds) != 2:
            raise FilterError(
                f"BETWEEN on {column!r} needs exactly 2 bounds, got {bounds!r}"
            )
        return f"{column}::numeric BETWEEN %s AND %s", [bounds[0], bounds[1]]

    if operator == "CLARIFICATION_NEEDED":
        raise FilterError(
            f"Filter on {column!r} is unresolved (CLARIFICATION_NEEDED); "
            "a valid route should not reach execution with this operator."
        )

    raise FilterError(f"Unsupported filter operator {operator!r} on {column!r}")


def build_where(resolved_filters: dict[str, Any]) -> tuple[str, list[Any]]:
    """Build a combined ``(where_sql, params)`` from all resolved filters.

    Returns an empty string and no params when there are no filters. Predicates
    are AND-ed together (cross-field semantics); OR semantics live *within* a
    single field via the OR_* operators.
    """
    clauses: list[str] = []
    params: list[Any] = []

    for column, spec in (resolved_filters or {}).items():
        if not isinstance(spec, dict):
            raise FilterError(
                f"Filter for {column!r} must be a dict, got {type(spec).__name__}"
            )
        clause, clause_params = build_predicate(column, spec)
        clauses.append(clause)
        params.extend(clause_params)

    where_sql = " AND ".join(clauses)
    return where_sql, params
