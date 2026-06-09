"""PostGIS executor for routed geospatial questions.

All spatial predicates and distances run in PostgreSQL/PostGIS. The executor
supports radius searches around companies, counties, places, and coordinates;
county polygon containment; and filtered map/list queries over geocoded rows.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from ..db import get_connection
from ..filters import build_where
from ..schemas import STATUS_FAILED, STATUS_SUCCESS, ExecutionResult
from .structured_sql import format_sql_for_display

logger = logging.getLogger(__name__)

DEFAULT_RADIUS_MILES = 50.0
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
_METERS_PER_MILE = 1609.344
_COORD_RE = re.compile(
    r"(?<![\d.])(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)(?![\d.])"
)
_RADIUS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(km|kilometers?|miles?|mi)\b",
    re.IGNORECASE,
)
_CLOSEST_RE = re.compile(r"\b(closest|nearest)\b", re.IGNORECASE)
_COUNTY_RE = re.compile(r"\b([A-Za-z][A-Za-z .'-]*?)\s+County\b", re.IGNORECASE)
_PROXIMITY_RE = re.compile(
    r"\b(near|nearby|closest|within|radius|distance|around|close to|proximity)\b",
    re.IGNORECASE,
)
_PROXIMITY_PLACE_RE = re.compile(
    r"(?:\bnear\b|\baround\b|\bclosest\s+to\b|"
    r"\bwithin\s+\d+(?:\.\d+)?\s*(?:km|kilometers?|miles?|mi)\s+of\b)"
    r"\s+([A-Za-z][A-Za-z .'-]+?)(?:[?.!,]|$)",
    re.IGNORECASE,
)

_SELECT_COLS = (
    "c.company, c.updated_location, c.ev_supply_chain_role, c.category, "
    "c.product_service, c.primary_oems, c.latitude, c.longitude"
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

_COMPANY_RESOLVE_SQL = """
SELECT company
FROM parent_chunks
WHERE geo IS NOT NULL
  AND (lower(company) = lower(%s) OR company ILIKE %s)
ORDER BY CASE WHEN lower(company) = lower(%s) THEN 0 ELSE 1 END,
         length(company),
         company
LIMIT 1;
"""
_COUNTY_EXISTS_SQL = (
    "SELECT 1 FROM georgia_counties WHERE county_name ILIKE %s LIMIT 1;"
)


def _fetch(
    sql: str,
    params: Sequence[Any] | dict[str, Any],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
    finally:
        conn.close()
    records = [dict(zip(cols, row)) for row in rows]
    if records and "distance_miles" in records[0]:
        records.sort(
            key=lambda row: (
                row.get("distance_miles")
                if row.get("distance_miles") is not None
                else 1e9
            )
        )
    return records[:limit] if limit is not None else records


def nearby_by_company(name: str, radius_miles: float, limit: int) -> list[dict[str, Any]]:
    """Return nearby companies using PostGIS; reused by disruption analysis."""
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


def _exists(sql: str, value: str) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (value,))
            return cur.fetchone() is not None
    finally:
        conn.close()


def _resolve_company_name(name: str) -> str | None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_COMPANY_RESOLVE_SQL, (name, f"%{name}%", name))
            row = cur.fetchone()
            return str(row[0]) if row else None
    finally:
        conn.close()


def _company_has_geo(name: str) -> bool:
    return _resolve_company_name(name) is not None


def _county_exists(name: str) -> bool:
    return _exists(_COUNTY_EXISTS_SQL, name)


def _extract_coordinates(question: str) -> tuple[float, float] | None:
    for match in _COORD_RE.finditer(question):
        lat, lon = float(match.group(1)), float(match.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    return None


def _extract_radius(final_route: dict[str, Any]) -> float:
    """Best-effort radius in miles from question/filters; default 50 miles."""
    question = str(final_route.get("question") or "")
    match = _RADIUS_RE.search(question)
    if match:
        value = float(match.group(1))
        return value / 1.609344 if match.group(2).lower().startswith(("km", "kilo")) else value

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
    try:
        value = int(final_route.get("limit"))
        return min(value, MAX_LIMIT) if value > 0 else DEFAULT_LIMIT
    except (TypeError, ValueError):
        return DEFAULT_LIMIT


def _values(value: Any) -> list[str]:
    items = value if isinstance(value, (list, tuple)) else [value]
    return [str(item).strip() for item in items if str(item or "").strip()]


def _location_candidates(final_route: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for field, spec in (final_route.get("resolved_filters") or {}).items():
        if field not in {"updated_location", "state"} or not isinstance(spec, dict):
            continue
        candidates.extend(_values(spec.get("value")))
    for raw_filter in final_route.get("raw_filters") or []:
        if not isinstance(raw_filter, dict):
            continue
        hint = str(raw_filter.get("field_hint") or "").lower()
        if any(token in hint for token in ("location", "county", "city", "state")):
            candidates.extend(_values(raw_filter.get("raw_value")))
    return list(dict.fromkeys(candidates))


def _county_anchor(final_route: dict[str, Any]) -> str | None:
    question = str(final_route.get("question") or "")
    candidates = [*final_route.get("entities", []), *_location_candidates(final_route)]
    candidates.extend(match.group(1).strip() for match in _COUNTY_RE.finditer(question))
    for value in candidates:
        county = re.sub(r"\s+County$", "", str(value).strip(), flags=re.IGNORECASE)
        if county and _county_exists(county):
            return county
    return None


def _place_anchor(final_route: dict[str, Any], county: str | None) -> str | None:
    ignored = {"georgia", "ga", "united states", "usa"}
    for value in _location_candidates(final_route):
        clean = re.sub(r"\s+County$", "", value, flags=re.IGNORECASE).strip()
        if clean and clean.casefold() not in ignored and clean.casefold() != str(county or "").casefold():
            return clean
    match = _PROXIMITY_PLACE_RE.search(str(final_route.get("question") or ""))
    if match:
        return match.group(1).strip()
    return None


def _filters_without(
    final_route: dict[str, Any],
    excluded: set[str] | None = None,
) -> dict[str, Any]:
    excluded = excluded or set()
    return {
        field: spec
        for field, spec in (final_route.get("resolved_filters") or {}).items()
        if field not in excluded
    }


def _candidate_cte(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    where_sql, params = build_where(filters)
    where = f" AND {where_sql}" if where_sql else ""
    return f"SELECT * FROM parent_chunks WHERE geo IS NOT NULL{where}", params


def _coordinate_query(
    lat: float,
    lon: float,
    radius_miles: float,
    filters: dict[str, Any],
    limit: int,
) -> tuple[str, list[Any]]:
    candidates, filter_params = _candidate_cte(filters)
    sql = f"""
WITH center AS (
    SELECT ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography AS geo
), candidates AS (
    {candidates}
)
SELECT {_SELECT_COLS},
       ST_Distance(c.geo, center.geo) / {_METERS_PER_MILE} AS distance_miles
FROM candidates c
JOIN center ON TRUE
WHERE ST_DWithin(c.geo, center.geo, %s)
ORDER BY distance_miles, c.company
LIMIT {limit};
"""
    return sql, [lon, lat, *filter_params, radius_miles * _METERS_PER_MILE]


def _company_query(
    company: str,
    radius_miles: float,
    filters: dict[str, Any],
    limit: int,
) -> tuple[str, list[Any]]:
    candidates, filter_params = _candidate_cte(filters)
    sql = f"""
WITH center AS (
    SELECT company, geo
    FROM parent_chunks
    WHERE lower(company) = lower(%s)
      AND geo IS NOT NULL
    LIMIT 1
), candidates AS (
    {candidates}
), ranked AS (
    SELECT DISTINCT ON (c.company)
           {_SELECT_COLS},
           ST_Distance(c.geo, center.geo) / {_METERS_PER_MILE} AS distance_miles
    FROM candidates c
    JOIN center ON TRUE
    WHERE lower(c.company) <> lower(center.company)
      AND ST_DWithin(c.geo, center.geo, %s)
    ORDER BY c.company, distance_miles
)
SELECT *
FROM ranked
ORDER BY distance_miles, company
LIMIT {limit};
"""
    return sql, [company, *filter_params, radius_miles * _METERS_PER_MILE]


def _closest_to_company_query(
    company: str,
    filters: dict[str, Any],
    limit: int,
) -> tuple[str, list[Any]]:
    candidates, filter_params = _candidate_cte(filters)
    sql = f"""
WITH center AS (
    SELECT company, geo
    FROM parent_chunks
    WHERE lower(company) = lower(%s)
      AND geo IS NOT NULL
    LIMIT 1
), candidates AS (
    {candidates}
)
SELECT {_SELECT_COLS},
       ST_Distance(c.geo, center.geo) / {_METERS_PER_MILE} AS distance_miles
FROM candidates c
JOIN center ON TRUE
WHERE lower(c.company) <> lower(center.company)
ORDER BY distance_miles, c.company
LIMIT {limit};
"""
    return sql, [company, *filter_params]


def _distance_to_company_query(
    center_company: str,
    target_companies: list[str],
    limit: int,
) -> tuple[str, list[Any]]:
    sql = f"""
WITH center AS (
    SELECT company, geo
    FROM parent_chunks
    WHERE lower(company) = lower(%s)
      AND geo IS NOT NULL
    LIMIT 1
), targets AS (
    SELECT DISTINCT ON (company)
           company, updated_location, ev_supply_chain_role, category,
           product_service, primary_oems, latitude, longitude, geo
    FROM parent_chunks
    WHERE geo IS NOT NULL
      AND lower(company) = ANY(%s)
    ORDER BY company
)
SELECT t.company, t.updated_location, t.ev_supply_chain_role, t.category,
       t.product_service, t.primary_oems, t.latitude, t.longitude,
       center.company AS distance_to,
       ST_Distance(t.geo, center.geo) / {_METERS_PER_MILE} AS distance_miles
FROM targets t
JOIN center ON TRUE
WHERE lower(t.company) <> lower(center.company)
ORDER BY distance_miles, t.company
LIMIT {limit};
"""
    return sql, [center_company, [name.casefold() for name in target_companies]]


def _nearby_targets_query(
    center_company: str,
    target_companies: list[str],
    radius_miles: float,
    limit: int,
) -> tuple[str, list[Any]]:
    sql = f"""
WITH center AS (
    SELECT company, geo
    FROM parent_chunks
    WHERE lower(company) = lower(%s)
      AND geo IS NOT NULL
    LIMIT 1
), targets AS (
    SELECT DISTINCT ON (company)
           company, updated_location, ev_supply_chain_role, category,
           product_service, primary_oems, latitude, longitude, geo
    FROM parent_chunks
    WHERE geo IS NOT NULL
      AND lower(company) = ANY(%s)
    ORDER BY company
)
SELECT t.company, t.updated_location, t.ev_supply_chain_role, t.category,
       t.product_service, t.primary_oems, t.latitude, t.longitude,
       center.company AS distance_to,
       ST_Distance(t.geo, center.geo) / {_METERS_PER_MILE} AS distance_miles
FROM targets t
JOIN center ON TRUE
WHERE lower(t.company) <> lower(center.company)
  AND ST_DWithin(t.geo, center.geo, %s)
ORDER BY distance_miles, t.company
LIMIT {limit};
"""
    return sql, [
        center_company,
        [name.casefold() for name in target_companies],
        radius_miles * _METERS_PER_MILE,
    ]


def _county_radius_query(
    county: str,
    radius_miles: float,
    filters: dict[str, Any],
    limit: int,
) -> tuple[str, list[Any]]:
    candidates, filter_params = _candidate_cte(filters)
    sql = f"""
WITH center AS (
    SELECT county_name, center_geo
    FROM georgia_counties
    WHERE lower(county_name) = lower(%s)
    LIMIT 1
), candidates AS (
    {candidates}
)
SELECT {_SELECT_COLS},
       ST_Distance(c.geo, center.center_geo) / {_METERS_PER_MILE} AS distance_miles
FROM candidates c
JOIN center ON TRUE
WHERE ST_DWithin(c.geo, center.center_geo, %s)
ORDER BY distance_miles, c.company
LIMIT {limit};
"""
    return sql, [county, *filter_params, radius_miles * _METERS_PER_MILE]


def _place_query(
    place: str,
    radius_miles: float,
    filters: dict[str, Any],
    limit: int,
) -> tuple[str, list[Any]]:
    candidates, filter_params = _candidate_cte(filters)
    sql = f"""
WITH center AS (
    SELECT ST_Centroid(ST_Collect(geom))::geography AS geo
    FROM parent_chunks
    WHERE geo IS NOT NULL AND updated_location ILIKE %s
), candidates AS (
    {candidates}
)
SELECT {_SELECT_COLS},
       ST_Distance(c.geo, center.geo) / {_METERS_PER_MILE} AS distance_miles
FROM candidates c
JOIN center ON center.geo IS NOT NULL
WHERE ST_DWithin(c.geo, center.geo, %s)
ORDER BY distance_miles, c.company
LIMIT {limit};
"""
    return sql, [f"%{place}%", *filter_params, radius_miles * _METERS_PER_MILE]


def _county_containment_query(
    county: str,
    filters: dict[str, Any],
    limit: int,
) -> tuple[str, list[Any]]:
    candidates, filter_params = _candidate_cte(filters)
    sql = f"""
WITH candidates AS (
    {candidates}
)
SELECT {_SELECT_COLS}, g.county_name AS county
FROM candidates c
JOIN georgia_counties g ON ST_Covers(g.geom, c.geom)
WHERE lower(g.county_name) = lower(%s)
ORDER BY c.company
LIMIT {limit};
"""
    return sql, [*filter_params, county]


def _filtered_points_query(
    filters: dict[str, Any],
    limit: int,
) -> tuple[str, list[Any]]:
    candidates, params = _candidate_cte(filters)
    sql = f"""
WITH candidates AS (
    {candidates}
)
SELECT {_SELECT_COLS}
FROM candidates c
ORDER BY c.company
LIMIT {limit};
"""
    return sql, params


def _company_center_filters(final_route: dict[str, Any]) -> dict[str, Any]:
    """Keep candidate filters while removing fields used only to identify the center."""
    filters = _filters_without(final_route, {"company", "updated_location"})
    question = str(final_route.get("question") or "").casefold()
    relationship_wording = any(
        token in question
        for token in ("linked to", "supplies", "support", "customer", "primary oem", "oem")
    )
    if not relationship_wording:
        filters.pop("primary_oems", None)
    return filters


def _format_rows(label: str, rows: list[dict[str, Any]], radius: float | None = None) -> str:
    if not rows:
        return f"No geocoded companies found for {label}."
    heading = (
        f"Companies within {radius:.0f} miles of {label}:"
        if radius is not None
        else f"Geocoded companies for {label}:"
    )
    lines = [heading]
    for index, row in enumerate(rows, start=1):
        distance = row.get("distance_miles")
        distance_text = f" ({distance:.1f} mi)" if distance is not None else ""
        details = [str(row.get("updated_location") or "").strip()]
        primary_oems = str(row.get("primary_oems") or "").strip()
        if primary_oems:
            details.append(f"Primary OEMs: {primary_oems}")
        lines.append(
            f"{index}. {row.get('company')}{distance_text} - "
            f"{'; '.join(detail for detail in details if detail)}"
        )
    return "\n".join(lines)


def _format_distances(center: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"No geocoded target companies were found for distance to {center}."
    lines = [f"Distances to {center}:"]
    for index, row in enumerate(rows, start=1):
        distance = row.get("distance_miles")
        distance_text = f"{distance:.1f} miles" if distance is not None else "unknown"
        lines.append(
            f"{index}. {row.get('company')}: {distance_text} - "
            f"{row.get('updated_location') or ''}"
        )
    return "\n".join(lines)


def _format_context_nearby(center: str, rows: list[dict[str, Any]], radius: float) -> str:
    if not rows:
        return (
            f"None of the listed companies are within {radius:.0f} miles of "
            f"{center.rstrip('.')}."
        )
    return _format_rows(center, rows, radius)


def _sql_command(
    label: str,
    sql: str,
    params: Sequence[Any] | dict[str, Any],
) -> dict[str, Any]:
    return {
        "label": label,
        "sql": sql,
        "sql_params": params,
        "sql_display": format_sql_for_display(sql, params),
    }


def _success(
    *,
    label: str,
    kind: str,
    rows: list[dict[str, Any]],
    sql_label: str,
    sql: str,
    params: Sequence[Any] | dict[str, Any],
    radius: float | None = None,
    center: dict[str, Any] | None = None,
) -> ExecutionResult:
    evidence: dict[str, Any] = {
        "type": "geo_results",
        "spatial_backend": "PostGIS",
        "spatial_operation": kind,
        "rows": rows,
        "sql_commands": [_sql_command(sql_label, sql, params)],
    }
    if center:
        evidence["center"] = center
    if radius is not None:
        evidence["radius_miles"] = radius
    return ExecutionResult(
        route="geo_search",
        status=STATUS_SUCCESS,
        answer=_format_rows(label, rows, radius),
        evidence=evidence,
    )


def execute_geo_search(final_route: dict[str, Any]) -> ExecutionResult:
    """Execute a validated geo route entirely through PostGIS."""
    question = str(final_route.get("question") or "")
    radius = _extract_radius(final_route)
    limit = _resolve_limit(final_route)
    entities = [
        str(entity).strip()
        for entity in (final_route.get("entities") or [])
        if str(entity).strip()
    ]
    context_entities = [
        str(entity).strip()
        for entity in (final_route.get("context_entities") or [])
        if str(entity).strip()
    ]
    operation = str(final_route.get("operation") or "").casefold()
    coordinates = _extract_coordinates(question)
    proximity = bool(coordinates or _PROXIMITY_RE.search(question))
    closest = bool(_CLOSEST_RE.search(question))
    explicit_radius = bool(_RADIUS_RE.search(question))

    if operation in {"distance_search", "nearby_search"} and entities and context_entities:
        center = _resolve_company_name(entities[-1])
        if center:
            if operation == "distance_search":
                sql, params = _distance_to_company_query(center, context_entities, limit)
                spatial_operation = "ST_Distance_company_targets"
                answer = None
                sql_label = "distance_to_company"
            else:
                sql, params = _nearby_targets_query(
                    center,
                    context_entities,
                    radius,
                    limit,
                )
                spatial_operation = "ST_DWithin_company_targets"
                sql_label = "nearby_context_companies"
            rows = _fetch(sql, params)
            if operation == "distance_search":
                answer = _format_distances(center, rows)
            else:
                answer = _format_context_nearby(center, rows, radius)
            return ExecutionResult(
                route="geo_search",
                status=STATUS_SUCCESS,
                answer=answer,
                evidence={
                    "type": "geo_results",
                    "spatial_backend": "PostGIS",
                    "spatial_operation": spatial_operation,
                    "center": {"kind": "company", "name": center},
                    **({"radius_miles": radius} if operation == "nearby_search" else {}),
                    "rows": rows,
                    "sql_commands": [
                        _sql_command(sql_label, sql, params)
                    ],
                },
            )

    county = _county_anchor(final_route)

    if coordinates:
        lat, lon = coordinates
        sql, params = _coordinate_query(
            lat,
            lon,
            radius,
            _filters_without(final_route, {"latitude", "longitude"}),
            limit,
        )
        rows = _fetch(sql, params)
        return _success(
            label=f"{lat}, {lon}",
            kind="ST_DWithin_coordinate",
            rows=rows,
            sql_label="nearby_by_coordinates",
            sql=sql,
            params=params,
            radius=radius,
            center={"kind": "coordinates", "latitude": lat, "longitude": lon},
        )

    if proximity:
        for name in entities:
            resolved_name = _resolve_company_name(name)
            if resolved_name:
                candidate_filters = _company_center_filters(final_route)
                if closest and not explicit_radius:
                    closest_limit = limit if final_route.get("limit") else 10
                    sql, params = _closest_to_company_query(
                        resolved_name,
                        candidate_filters,
                        closest_limit,
                    )
                    rows = _fetch(sql, params)
                    return ExecutionResult(
                        route="geo_search",
                        status=STATUS_SUCCESS,
                        answer=_format_distances(resolved_name, rows),
                        evidence={
                            "type": "geo_results",
                            "spatial_backend": "PostGIS",
                            "spatial_operation": "ST_Distance_closest_company",
                            "center": {"kind": "company", "name": resolved_name},
                            "rows": rows,
                            "sql_commands": [
                                _sql_command("closest_to_company", sql, params)
                            ],
                        },
                    )
                sql, params = _company_query(
                    resolved_name,
                    radius,
                    candidate_filters,
                    limit,
                )
                rows = _fetch(sql, params)
                return _success(
                    label=resolved_name,
                    kind="ST_DWithin_company",
                    rows=rows,
                    sql_label="nearby_by_company",
                    sql=sql,
                    params=params,
                    radius=radius,
                    center={"kind": "company", "name": resolved_name},
                )

        if county:
            sql, params = _county_radius_query(
                county,
                radius,
                _filters_without(final_route, {"updated_location"}),
                limit,
            )
            rows = _fetch(sql, params)
            return _success(
                label=f"{county} County",
                kind="ST_DWithin_county_center",
                rows=rows,
                sql_label="nearby_by_county",
                sql=sql,
                params=params,
                radius=radius,
                center={"kind": "county", "name": county},
            )

        place = _place_anchor(final_route, county)
        if place:
            sql, params = _place_query(
                place,
                radius,
                _filters_without(final_route, {"updated_location"}),
                limit,
            )
            rows = _fetch(sql, params)
            return _success(
                label=place,
                kind="ST_DWithin_place_centroid",
                rows=rows,
                sql_label="nearby_by_place",
                sql=sql,
                params=params,
                radius=radius,
                center={"kind": "place", "name": place},
            )

    if county:
        sql, params = _county_containment_query(
            county,
            _filters_without(final_route, {"updated_location"}),
            limit,
        )
        rows = _fetch(sql, params)
        return _success(
            label=f"{county} County",
            kind="ST_Covers_county",
            rows=rows,
            sql_label="companies_in_county",
            sql=sql,
            params=params,
            center={"kind": "county", "name": county},
        )

    filters = _filters_without(final_route)
    if filters or re.search(r"\b(map|geospatial|spatial)\b", question, re.IGNORECASE):
        sql, params = _filtered_points_query(filters, limit)
        rows = _fetch(sql, params)
        return _success(
            label="the requested filters",
            kind="PostGIS_geocoded_filter",
            rows=rows,
            sql_label="geocoded_filtered_companies",
            sql=sql,
            params=params,
        )

    return ExecutionResult(
        route="geo_search",
        status=STATUS_FAILED,
        answer="",
        evidence={"type": "error", "spatial_backend": "PostGIS"},
        error=(
            "geo_search could not resolve coordinates, a company/place/county "
            "anchor, or structured filters."
        ),
    )
