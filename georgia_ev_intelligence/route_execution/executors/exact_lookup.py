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

logger = logging.getLogger(__name__)

TABLE = "parent_chunks"
_MAX_ROWS = 50


def _output_columns(final_route: dict[str, Any]) -> list[str]:
    requested = [ensure_allowed(c) for c in (final_route.get("requested_columns") or [])]
    if requested:
        return list(dict.fromkeys(["company", *requested]))
    return list(DEFAULT_COLUMNS)


def _resolve_rows(name: str, columns: list[str]) -> list[dict[str, Any]]:
    select_cols = ", ".join(columns)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # 1) exact, case-insensitive match.
            cur.execute(
                f"SELECT {select_cols} FROM {TABLE} WHERE lower(company) = lower(%s) "
                f"ORDER BY company LIMIT %s;",
                (name, _MAX_ROWS),
            )
            rows = cur.fetchall()
            # 2) fall back to substring match.
            if not rows:
                cur.execute(
                    f"SELECT {select_cols} FROM {TABLE} WHERE company ILIKE %s "
                    f"ORDER BY company LIMIT %s;",
                    (f"%{name}%", _MAX_ROWS),
                )
                rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]
    finally:
        conn.close()
    return [dict(zip(col_names, row)) for row in rows]


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
    records = _resolve_rows(name, columns)

    if not records:
        return ExecutionResult(
            route="exact_lookup",
            status=STATUS_SUCCESS,
            answer=f"I could not find a company matching '{name}' in the knowledge base.",
            evidence={"type": "clarification", "reason": "company not resolved", "query": name},
        )

    return ExecutionResult(
        route="exact_lookup",
        status=STATUS_SUCCESS,
        answer=fmt.format_structured_rows(records, columns),
        evidence={"type": "structured_rows", "rows": records, "columns": columns, "query": name},
    )
