"""Exact lookup executor: resolve a named company and fetch its fields.

Flow (execution README §10): entity name -> company resolver -> rows. Resolution
tries an exact (case-insensitive) company match first, then a substring match.
If the company cannot be resolved, a clarification-style result is returned
instead of guessing.
"""
from __future__ import annotations

import logging
from typing import Any

from .. import answer_formatter as fmt
from ..column_allowlist import DEFAULT_COLUMNS, ensure_allowed
from ..db import get_connection
from ..schemas import STATUS_SUCCESS, ExecutionResult
from .structured_sql import format_sql_for_display

logger = logging.getLogger(__name__)

TABLE = "parent_chunks"
_MAX_ROWS = 50


def _output_columns(final_route: dict[str, Any]) -> list[str]:
    requested = [ensure_allowed(c) for c in (final_route.get("requested_columns") or [])]
    if requested:
        return list(dict.fromkeys(["company", *requested]))
    return list(DEFAULT_COLUMNS)


def _resolve_rows(
    name: str,
    columns: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    select_cols = ", ".join(columns)
    sql_commands: list[dict[str, Any]] = []
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # 1) exact, case-insensitive match.
            exact_sql = (
                f"SELECT {select_cols} FROM {TABLE} WHERE lower(company) = lower(%s) "
                f"ORDER BY company LIMIT %s;"
            )
            exact_params = [name, _MAX_ROWS]
            cur.execute(
                exact_sql,
                exact_params,
            )
            sql_commands.append(_sql_command("exact_company_match", exact_sql, exact_params))
            rows = cur.fetchall()
            # 2) fall back to substring match.
            if not rows:
                substring_sql = (
                    f"SELECT {select_cols} FROM {TABLE} WHERE company ILIKE %s "
                    f"ORDER BY company LIMIT %s;"
                )
                substring_params = [f"%{name}%", _MAX_ROWS]
                cur.execute(
                    substring_sql,
                    substring_params,
                )
                sql_commands.append(
                    _sql_command("substring_company_match", substring_sql, substring_params)
                )
                rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]
    finally:
        conn.close()
    return [dict(zip(col_names, row)) for row in rows], sql_commands


def _sql_command(label: str, sql: str, params: list[Any]) -> dict[str, Any]:
    return {
        "label": label,
        "sql": sql,
        "sql_params": params,
        "sql_display": format_sql_for_display(sql, params),
    }


def execute_exact_lookup(final_route: dict[str, Any]) -> ExecutionResult:
    entities = final_route.get("entities") or []
    name = next((str(e).strip() for e in entities if str(e).strip()), "")

    if not name:
        return ExecutionResult(
            route="exact_lookup",
            status=STATUS_SUCCESS,
            answer="I need the name of the company to look up.",
            evidence={"type": "clarification", "reason": "no entity provided"},
        )

    columns = _output_columns(final_route)
    records, sql_commands = _resolve_rows(name, columns)

    if not records:
        return ExecutionResult(
            route="exact_lookup",
            status=STATUS_SUCCESS,
            answer=f"I could not find a company matching '{name}' in the knowledge base.",
            evidence={
                "type": "clarification",
                "reason": "company not resolved",
                "query": name,
                "sql_commands": sql_commands,
            },
        )

    return ExecutionResult(
        route="exact_lookup",
        status=STATUS_SUCCESS,
        answer=fmt.format_structured_rows(records, columns),
        evidence={
            "type": "structured_rows",
            "rows": records,
            "columns": columns,
            "query": name,
            "sql_commands": sql_commands,
        },
    )
