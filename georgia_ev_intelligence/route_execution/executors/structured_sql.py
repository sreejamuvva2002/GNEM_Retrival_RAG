"""Structured SQL executor over the ``parent_chunks`` table.

Builds a safe parameterized SELECT/COUNT/GROUP BY from the validated route's
``resolved_filters``, ``requested_columns``, ``group_by``, ``sort_by`` and
``limit``. The LLM never writes SQL: every identifier is allowlist-checked and
every value is a bound parameter. Query construction is split from execution so
it can be unit-tested without a database.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from .. import answer_formatter as fmt
from ..column_allowlist import DEFAULT_COLUMNS, ensure_allowed
from ..db import get_connection
from ..filters import build_where
from ..schemas import STATUS_SUCCESS, ExecutionResult

logger = logging.getLogger(__name__)

TABLE = "parent_chunks"
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000

# Operations that count rather than list.
_COUNT_OPS = {"count_records"}
_GROUP_OPS = {"group_records", "aggregate_records"}

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
        group_by = _validated_columns(final_route.get("group_by") or [])
        if not group_by:
            # Fall back to a plain count if the router asked to group by nothing.
            sql = f"SELECT COUNT(*) AS count FROM {TABLE}{where_clause};"
            return sql, params, ["count"], "count"
        group_cols = ", ".join(group_by)
        sql = (
            f"SELECT {group_cols}, COUNT(*) AS count FROM {TABLE}{where_clause} "
            f"GROUP BY {group_cols} ORDER BY count DESC;"
        )
        return sql, params, [*group_by, "count"], "group"

    # Default: list_records.
    requested = _validated_columns(final_route.get("requested_columns") or [])
    columns = list(dict.fromkeys(["company", *requested])) if requested else list(DEFAULT_COLUMNS)

    select_cols = ", ".join(columns)
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
            return _hybrid_fallback(final_route, sql, params)
        group_by = [c for c in columns if c != "count"]
        return ExecutionResult(
            route="structured_sql",
            status=STATUS_SUCCESS,
            answer=fmt.format_group_counts(records, group_by),
            evidence={"type": "group_counts", "groups": records, **_sql_evidence(sql, params)},
        )

    if not records:
        # Structured filtering matched nothing — often the router put the value on
        # the wrong column or added noise words. Fall back to hybrid retrieval
        # (BM25 + dense) on the question so the answer still has real evidence.
        return _hybrid_fallback(final_route, sql, params)

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


def _hybrid_fallback(
    final_route: dict[str, Any],
    structured_sql: str,
    structured_params: list[Any],
) -> ExecutionResult:
    """Run BM25 + dense retrieval when structured filtering returned no rows.

    The result is tagged as a fallback (and keeps the originating ``structured_sql``)
    so the route label stays stable for downstream LLM grounding while the evidence
    reflects the document retrieval that actually produced it.
    """
    from .hybrid_search import execute_hybrid_search

    try:
        result = execute_hybrid_search(final_route)
    except Exception as exc:  # retrieval/DB unavailable -> honest empty result
        logger.warning("hybrid fallback failed: %s", exc)
        return ExecutionResult(
            route="structured_sql",
            status=STATUS_SUCCESS,
            answer="Found 0 matching records.",
            evidence={"type": "structured_rows", "rows": [], "columns": ["company"],
                      **_sql_evidence(structured_sql, structured_params)},
        )

    result.route = "structured_sql"
    if isinstance(result.evidence, dict):
        result.evidence["fallback"] = "hybrid_search"
        result.evidence["structured_sql"] = structured_sql
        result.evidence["structured_sql_params"] = structured_params
        result.evidence["structured_sql_display"] = format_sql_for_display(
            structured_sql,
            structured_params,
        )
    return result
