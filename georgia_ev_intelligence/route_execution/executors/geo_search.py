"""Geo search executor (PostGIS) over ``parent_chunks`` and ``georgia_counties``.

Supports radius searches centred on a company, a Georgia county, or raw
coordinates, using the PostGIS ``geo``/``center_geo`` columns created by
``scripts/setup_postgis.py``. Distances are reported in miles.

These helpers are also reused by the disruption executor to find nearby
alternatives, so the core queries are exposed as standalone functions.
"""
from __future__ import annotations

import logging
from typing import Any

from ..db import get_connection
from ..schemas import STATUS_FAILED, STATUS_SUCCESS, ExecutionResult
from .structured_sql import format_sql_for_display

logger = logging.getLogger(__name__)

DEFAULT_RADIUS_MILES = 50.0
DEFAULT_LIMIT = 25
_METERS_PER_MILE = 1609.344

# Result columns shared by the nearby queries.
_SELECT_COLS = (
    "c.company, c.updated_location, c.ev_supply_chain_role, c.category, "
    "c.latitude, c.longitude"
)

_NEARBY_BY_COMPANY_SQL = f"""
WITH center AS (
    SELECT record_id, company, geo
    FROM parent_chunks
    WHERE lower(company) = lower(%(name)s)
      AND geo IS NOT NULL
    LIMIT 1
)
SELECT DISTINCT ON (c.company)
    {_SELECT_COLS},
    ST_Distance(c.geo, center.geo) / {_METERS_PER_MILE} AS distance_miles
FROM parent_chunks c
JOIN center ON TRUE
WHERE c.geo IS NOT NULL
  AND lower(c.company) <> lower(center.company)
  AND ST_DWithin(c.geo, center.geo, %(radius_m)s)
ORDER BY c.company, distance_miles ASC;
"""

_NEARBY_BY_COUNTY_SQL = f"""
WITH center AS (
    SELECT county_name, center_geo
    FROM georgia_counties
    WHERE county_name ILIKE %(name)s
    LIMIT 1
)
SELECT DISTINCT ON (c.company)
    {_SELECT_COLS},
    ST_Distance(c.geo, center.center_geo) / {_METERS_PER_MILE} AS distance_miles
FROM parent_chunks c
JOIN center ON TRUE
WHERE c.geo IS NOT NULL
  AND ST_DWithin(c.geo, center.center_geo, %(radius_m)s)
ORDER BY c.company, distance_miles ASC;
"""

_COMPANY_HAS_GEO_SQL = (
    "SELECT 1 FROM parent_chunks WHERE lower(company) = lower(%s) "
    "AND geo IS NOT NULL LIMIT 1;"
)
_COUNTY_EXISTS_SQL = (
    "SELECT 1 FROM georgia_counties WHERE county_name ILIKE %s LIMIT 1;"
)


def _fetch(sql: str, params: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
    finally:
        conn.close()
    records = [dict(zip(cols, row)) for row in rows]
    records.sort(key=lambda r: r.get("distance_miles") if r.get("distance_miles") is not None else 1e9)
    return records[:limit]


def nearby_by_company(name: str, radius_miles: float, limit: int) -> list[dict[str, Any]]:
    return _fetch(
        _NEARBY_BY_COMPANY_SQL,
        {"name": name, "radius_m": radius_miles * _METERS_PER_MILE},
        limit,
    )


def nearby_by_county(county: str, radius_miles: float, limit: int) -> list[dict[str, Any]]:
    return _fetch(
        _NEARBY_BY_COUNTY_SQL,
        {"name": county, "radius_m": radius_miles * _METERS_PER_MILE},
        limit,
    )


def _company_has_geo(name: str) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_COMPANY_HAS_GEO_SQL, (name,))
            return cur.fetchone() is not None
    finally:
        conn.close()


def _county_exists(name: str) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_COUNTY_EXISTS_SQL, (name,))
            return cur.fetchone() is not None
    finally:
        conn.close()


def _extract_radius(final_route: dict[str, Any]) -> float:
    """Best-effort radius (miles) from resolved filters; default 50."""
    for key, spec in (final_route.get("resolved_filters") or {}).items():
        if not isinstance(spec, dict):
            continue
        if any(token in key.lower() for token in ("radius", "distance", "mile")):
            try:
                return float(spec.get("value"))
            except (TypeError, ValueError):
                continue
    return DEFAULT_RADIUS_MILES


def _resolve_limit(final_route: dict[str, Any]) -> int:
    limit = final_route.get("limit")
    try:
        value = int(limit)
        return value if value > 0 else DEFAULT_LIMIT
    except (TypeError, ValueError):
        return DEFAULT_LIMIT


def _format_nearby(center_label: str, radius: float, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"No companies found within {radius:.0f} miles of {center_label}."
    lines = [f"Companies within {radius:.0f} miles of {center_label}:"]
    for idx, row in enumerate(rows, start=1):
        dist = row.get("distance_miles")
        dist_txt = f"{dist:.1f} mi" if dist is not None else "?"
        lines.append(f"{idx}. {row.get('company')} ({dist_txt}) — {row.get('updated_location') or ''}")
    return "\n".join(lines)


def _nearby_sql_command(
    label: str,
    sql: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "label": label,
        "sql": sql,
        "sql_params": params,
        "sql_display": format_sql_for_display(sql, params),
    }


def execute_geo_search(final_route: dict[str, Any]) -> ExecutionResult:
    radius = _extract_radius(final_route)
    limit = _resolve_limit(final_route)
    entities = [str(e).strip() for e in (final_route.get("entities") or []) if str(e).strip()]

    # 1) Prefer a company centre.
    for name in entities:
        if _company_has_geo(name):
            rows = nearby_by_company(name, radius, limit)
            params = {"name": name, "radius_m": radius * _METERS_PER_MILE}
            return ExecutionResult(
                route="geo_search",
                status=STATUS_SUCCESS,
                answer=_format_nearby(name, radius, rows),
                evidence={
                    "type": "geo_results",
                    "center": {"kind": "company", "name": name},
                    "radius_miles": radius,
                    "rows": rows,
                    "sql_commands": [
                        _nearby_sql_command(
                            "nearby_by_company",
                            _NEARBY_BY_COMPANY_SQL,
                            params,
                        )
                    ],
                },
            )

    # 2) Fall back to a county centre.
    for name in entities:
        county = name.replace("County", "").strip()
        if _county_exists(county):
            rows = nearby_by_county(county, radius, limit)
            params = {"name": county, "radius_m": radius * _METERS_PER_MILE}
            return ExecutionResult(
                route="geo_search",
                status=STATUS_SUCCESS,
                answer=_format_nearby(f"{county} County", radius, rows),
                evidence={
                    "type": "geo_results",
                    "center": {"kind": "county", "name": county},
                    "radius_miles": radius,
                    "rows": rows,
                    "sql_commands": [
                        _nearby_sql_command(
                            "nearby_by_county",
                            _NEARBY_BY_COUNTY_SQL,
                            params,
                        )
                    ],
                },
            )

    return ExecutionResult(
        route="geo_search",
        status=STATUS_FAILED,
        answer="",
        evidence={"type": "error"},
        error=(
            "geo_search could not resolve a centre (no matching company with "
            f"coordinates or county among entities={entities!r})."
        ),
    )
