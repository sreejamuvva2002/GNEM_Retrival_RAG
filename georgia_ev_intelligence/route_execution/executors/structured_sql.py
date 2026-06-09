"""Structured SQL executor over the ``parent_chunks`` table.

Builds a safe parameterized SELECT/COUNT/GROUP BY/aggregate query from the
validated route's ``resolved_filters``, ``requested_columns``, ``group_by``,
``sort_by`` and ``limit``. The LLM never writes SQL: every identifier is
allowlist-checked and every value is a bound parameter. Query construction is
split from execution so it can be unit-tested without a database.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from .. import answer_formatter as fmt
from ..column_allowlist import ALLOWED_COLUMNS, DEFAULT_COLUMNS, NUMERIC_COLUMNS, ensure_allowed
from ..db import get_connection
from ..filters import build_where
from ..schemas import STATUS_SUCCESS, ExecutionResult

TABLE = "parent_chunks"
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
# Coordinates fetched for every listed row so the UI can map it, but excluded
# from the reported display columns (see build_query).
MAP_COLUMNS = ("latitude", "longitude")

# Operations that count rather than list.
_COUNT_OPS = {"count_records"}
_GROUP_OPS = {"group_records", "aggregate_records"}
_COUNTY_EXPR = (
    "NULLIF(btrim((regexp_match(updated_location, '([^,]+ County)', 'i'))[1]), '')"
)

_NAMED_PLACEHOLDER = re.compile(r"%\(([^)]+)\)s")


def format_sql_for_display(sql: str, params: list[Any] | tuple[Any, ...] | dict[str, Any]) -> str:
    """Render a parameterized SQL query with readable literals for audit exports.

    This string is only for logs/XLSX inspection. Execution still uses the
    original parameterized SQL with bound values.
    """
    if not params:
        return sql

    if isinstance(params, dict):
        return _NAMED_PLACEHOLDER.sub(
            lambda match: _sql_literal(params.get(match.group(1))),
            sql,
        )

    pieces = sql.split("%s")
    if len(pieces) == 1:
        return sql

    rendered = [pieces[0]]
    for index, value in enumerate(params):
        rendered.append(_sql_literal(value))
        if index + 1 < len(pieces):
            rendered.append(pieces[index + 1])
    if len(pieces) > len(params) + 1:
        rendered.extend(pieces[len(params) + 1:])
    return "".join(rendered)


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, (date, datetime)):
        return _quote(value.isoformat())
    if isinstance(value, (list, tuple, set)):
        return "ARRAY[" + ", ".join(_sql_literal(item) for item in value) + "]"
    return _quote(str(value))


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_evidence(sql: str, params: list[Any]) -> dict[str, Any]:
    return {
        "sql": sql,
        "sql_params": params,
        "sql_display": format_sql_for_display(sql, params),
    }


def _validated_columns(columns: list[str]) -> list[str]:
    return [ensure_allowed(c) for c in columns]


def _order_terms(sort_by: list[str]) -> list[str]:
    """Parse ``["employment DESC", ...]`` into safe ``"<col> <DIR>"`` terms.

    The column is allowlist-checked; the direction is whitelisted to ASC/DESC
    (defaulting to ASC) so nothing route-supplied reaches the SQL verbatim.
    """
    terms: list[str] = []
    for raw in sort_by:
        parts = str(raw).split()
        if not parts:
            continue
        column = ensure_allowed(parts[0])
        direction = "DESC" if len(parts) > 1 and parts[1].upper() == "DESC" else "ASC"
        terms.append(f"{column} {direction}")
    return terms


def _group_terms(group_by: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return safe select terms, GROUP BY expressions, and output names."""
    select_terms: list[str] = []
    group_terms: list[str] = []
    output_names: list[str] = []
    for raw in group_by:
        if raw == "county":
            select_terms.append(f"{_COUNTY_EXPR} AS county")
            group_terms.append(_COUNTY_EXPR)
            output_names.append("county")
            continue
        column = ensure_allowed(raw)
        select_terms.append(column)
        group_terms.append(column)
        output_names.append(column)
    return select_terms, group_terms, output_names


def _with_virtual_group_filters(where_sql: str, group_by: list[str]) -> str:
    clauses = [where_sql] if where_sql else []
    if "county" in group_by:
        clauses.append(f"{_COUNTY_EXPR} IS NOT NULL")
    return " AND ".join(clauses)


def _aggregate_spec(final_route: dict[str, Any]) -> tuple[str, str] | None:
    """Return a safe ``(SQL expression, output alias)`` for numeric aggregation."""
    question = str(final_route.get("question") or "").lower()
    candidates: list[str] = []
    for raw in final_route.get("sort_by") or []:
        parts = str(raw).split()
        if parts:
            candidates.append(parts[0])
    candidates.extend(final_route.get("requested_columns") or [])
    if "employment" in question:
        candidates.append("employment")

    metric = next((field for field in candidates if field in NUMERIC_COLUMNS), None)
    if metric is None:
        return None

    if any(token in question for token in ("average", "avg", "mean")):
        function, prefix = "AVG", "average"
    elif any(token in question for token in ("minimum", "lowest", "smallest")):
        function, prefix = "MIN", "minimum"
    elif any(token in question for token in ("maximum", "highest", "largest")) and "total" not in question:
        function, prefix = "MAX", "maximum"
    else:
        function, prefix = "SUM", "total"
    return f"{function}({metric})", f"{prefix}_{metric}"


def _aggregate_order(final_route: dict[str, Any], alias: str) -> str:
    sort_by = final_route.get("sort_by") or []
    if not sort_by:
        return f" ORDER BY {alias} DESC"
    parts = str(sort_by[0]).split()
    direction = "ASC" if len(parts) > 1 and parts[1].upper() == "ASC" else "DESC"
    return f" ORDER BY {alias} {direction}"


def _resolve_limit(limit: Any) -> int:
    if limit is None:
        return DEFAULT_LIMIT
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    if value <= 0:
        return DEFAULT_LIMIT
    return min(value, MAX_LIMIT)


def build_query(final_route: dict[str, Any]) -> tuple[str, list[Any], list[str], str]:
    """Return ``(sql, params, output_columns, mode)`` for a structured route.

    ``mode`` is one of ``"list"``, ``"count"``, ``"group"`` and tells the caller
    how to interpret the result set. No database access happens here.
    """
    operation = (final_route.get("operation") or "list_records").lower()
    resolved_filters = final_route.get("resolved_filters") or {}
    where_sql, params = build_where(resolved_filters)
    where_clause = f" WHERE {where_sql}" if where_sql else ""

    if operation in _COUNT_OPS:
        sql = f"SELECT COUNT(*) AS count FROM {TABLE}{where_clause};"
        return sql, params, ["count"], "count"

    if operation in _GROUP_OPS:
        group_by = list(final_route.get("group_by") or [])
        if not group_by:
            # Fall back to a plain count if the router asked to group by nothing.
            sql = f"SELECT COUNT(*) AS count FROM {TABLE}{where_clause};"
            return sql, params, ["count"], "count"
        select_terms, group_terms, group_names = _group_terms(group_by)
        grouped_where = _with_virtual_group_filters(where_sql, group_by)
        grouped_where_clause = f" WHERE {grouped_where}" if grouped_where else ""
        select_cols = ", ".join(select_terms)
        group_cols = ", ".join(group_terms)

        if operation == "aggregate_records":
            aggregate = _aggregate_spec(final_route)
            if aggregate is not None:
                expression, alias = aggregate
                limit = _resolve_limit(final_route.get("limit"))
                sql = (
                    f"SELECT {select_cols}, {expression} AS {alias} FROM {TABLE}"
                    f"{grouped_where_clause} GROUP BY {group_cols}"
                    f"{_aggregate_order(final_route, alias)} LIMIT {limit};"
                )
                return sql, params, [*group_names, alias], "aggregate"

        sql = (
            f"SELECT {select_cols}, COUNT(*) AS count FROM {TABLE}{grouped_where_clause} "
            f"GROUP BY {group_cols} ORDER BY count DESC;"
        )
        return sql, params, [*group_names, "count"], "group"

    # Default: list_records.
    requested = _validated_columns(final_route.get("requested_columns") or [])
    # Surface the columns we filtered on so each returned row is self-describing
    # and the answer can cite *why* it matched (e.g. show ev_supply_chain_role
    # when the question filters by Battery Cell / Battery Pack). Without this, a
    # role filter paired with `requested_columns=['category']` selects only
    # company+category, and the grounded answer can't tell the matched roles
    # apart. Latitude/longitude are geo mechanics, not descriptive, so skip them.
    filter_cols = [
        col
        for col in (resolved_filters or {})
        if col in ALLOWED_COLUMNS and col not in {"latitude", "longitude"}
    ]
    if requested or filter_cols:
        columns = list(dict.fromkeys(["company", *requested, *filter_cols]))
    else:
        columns = list(DEFAULT_COLUMNS)

    # Always fetch coordinates so the UI can place each returned company on the
    # map, but keep them OUT of the reported `columns` so they are neither listed
    # in the answer text nor fed to the grounding LLM (which would otherwise
    # print raw lat/lon). The extra row keys are consumed only by the map / source
    # cards. Skip the duplicate if the user explicitly requested coordinates.
    select_columns = list(dict.fromkeys([*columns, *MAP_COLUMNS]))
    select_cols = ", ".join(select_columns)
    order_terms = _order_terms(final_route.get("sort_by") or [])
    order_clause = f" ORDER BY {', '.join(order_terms)}" if order_terms else " ORDER BY company"
    limit = _resolve_limit(final_route.get("limit"))

    sql = f"SELECT {select_cols} FROM {TABLE}{where_clause}{order_clause} LIMIT {limit};"
    return sql, params, columns, "list"


def execute_structured_sql(final_route: dict[str, Any]) -> ExecutionResult:
    sql, params, columns, mode = build_query(final_route)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]
    finally:
        conn.close()

    records = [dict(zip(col_names, row)) for row in rows]

    if mode == "count":
        count = int(records[0]["count"]) if records else 0
        return ExecutionResult(
            route="structured_sql",
            status=STATUS_SUCCESS,
            answer=fmt.format_count(count),
            evidence={"type": "count", "count": count, **_sql_evidence(sql, params)},
        )

    if mode == "group":
        if not records:
            return ExecutionResult(
                route="structured_sql",
                status=STATUS_SUCCESS,
                answer="No groups matched.",
                evidence={"type": "group_counts", "groups": [], **_sql_evidence(sql, params)},
            )
        group_by = [c for c in columns if c != "count"]
        return ExecutionResult(
            route="structured_sql",
            status=STATUS_SUCCESS,
            answer=fmt.format_group_counts(records, group_by),
            evidence={"type": "group_counts", "groups": records, **_sql_evidence(sql, params)},
        )

    if mode == "aggregate":
        aggregate_column = columns[-1]
        group_by = columns[:-1]
        return ExecutionResult(
            route="structured_sql",
            status=STATUS_SUCCESS,
            answer=fmt.format_group_aggregates(records, group_by, aggregate_column),
            evidence={
                "type": "group_aggregates",
                "groups": records,
                "group_by": group_by,
                "aggregate_column": aggregate_column,
                **_sql_evidence(sql, params),
            },
        )

    if not records:
        # A structured query's filters are authoritative. Returning unrelated
        # semantic-search contexts here can turn a zero-result query into a false
        # answer, so preserve the honest empty structured result.
        return ExecutionResult(
            route="structured_sql",
            status=STATUS_SUCCESS,
            answer="Found 0 matching records.",
            evidence={
                "type": "structured_rows",
                "rows": [],
                "columns": columns,
                **_sql_evidence(sql, params),
            },
        )

    return ExecutionResult(
        route="structured_sql",
        status=STATUS_SUCCESS,
        answer=fmt.format_structured_rows(records, columns),
        evidence={
            "type": "structured_rows",
            "rows": records,
            "columns": columns,
            **_sql_evidence(sql, params),
        },
    )
