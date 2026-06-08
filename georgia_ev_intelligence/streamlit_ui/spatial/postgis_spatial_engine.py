"""PostGIS-backed spatial engine used by the Streamlit map service."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd

from georgia_ev_intelligence.route_execution.db import get_connection

_METERS_PER_KM = 1000.0
_KM_TO_MILES = 0.621371

_COMPANY_COLUMNS = """
    c.record_id,
    c.company,
    c.category,
    c.industry_group,
    c.updated_location AS location,
    c.address,
    split_part(c.updated_location, ',', 1) AS city,
    county.county_name AS county,
    c.ev_supply_chain_role,
    c.primary_oems,
    c.supplier_or_affiliation_type,
    c.employment,
    c.product_service,
    c.ev_battery_relevant,
    c.primary_facility_type,
    c.latitude,
    c.longitude,
    'PostGIS'::text AS coordinate_source
"""

_COUNTY_LATERAL_JOIN = """
LEFT JOIN LATERAL (
    SELECT g.county_name
    FROM georgia_counties g
    WHERE ST_Covers(g.geom, c.geom)
    LIMIT 1
) county ON TRUE
"""


class PostGISSpatialEngine:
    """Serve map and geo-analysis data using PostgreSQL/PostGIS operations."""

    def __init__(self) -> None:
        self.companies_df = self._load_companies()
        counties = self._query_df(
            "SELECT county_name FROM georgia_counties ORDER BY county_name;"
        )
        self.county_names = (
            counties["county_name"].dropna().astype(str).tolist()
            if "county_name" in counties
            else []
        )

    @staticmethod
    def _query_df(sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, list(params or []))
                rows = cur.fetchall()
                columns = [description[0] for description in cur.description]
        finally:
            conn.close()
        return pd.DataFrame(rows, columns=columns)

    def _load_companies(self) -> pd.DataFrame:
        return self._query_df(
            f"""
SELECT {_COMPANY_COLUMNS}
FROM parent_chunks c
{_COUNTY_LATERAL_JOIN}
WHERE c.geo IS NOT NULL
ORDER BY c.company, c.updated_location;
"""
        )

    @staticmethod
    def _limit_to_candidates(
        rows: pd.DataFrame,
        candidates: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if candidates is None or candidates.empty or rows.empty:
            return rows
        if "record_id" in candidates and "record_id" in rows:
            return rows[rows["record_id"].isin(candidates["record_id"])].reset_index(drop=True)
        if "company" in candidates and "company" in rows:
            return rows[rows["company"].isin(candidates["company"])].reset_index(drop=True)
        return rows

    def resolve_place_coordinates(self, place_name: str) -> Optional[tuple[float, float]]:
        rows = self._query_df(
            """
WITH matches AS (
    SELECT 1 AS priority,
           ST_Y(center_geom) AS latitude,
           ST_X(center_geom) AS longitude
    FROM georgia_counties
    WHERE lower(county_name) = lower(regexp_replace(%s, '\\s+County$', '', 'i'))
      AND center_geom IS NOT NULL
    UNION ALL
    SELECT 2 AS priority,
           ST_Y(geom) AS latitude,
           ST_X(geom) AS longitude
    FROM parent_chunks
    WHERE lower(company) = lower(%s)
      AND geom IS NOT NULL
    UNION ALL
    SELECT 3 AS priority,
           ST_Y(ST_Centroid(ST_Collect(geom))) AS latitude,
           ST_X(ST_Centroid(ST_Collect(geom))) AS longitude
    FROM parent_chunks
    WHERE updated_location ILIKE %s
      AND geom IS NOT NULL
    HAVING COUNT(geom) > 0
)
SELECT latitude, longitude
FROM matches
ORDER BY priority
LIMIT 1;
""",
            [place_name, place_name, f"%{place_name}%"],
        )
        if rows.empty:
            return None
        return float(rows.iloc[0]["latitude"]), float(rows.iloc[0]["longitude"])

    def companies_in_counties(
        self,
        counties: Sequence[str],
        candidates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        wanted = [
            str(county).strip().removesuffix(" County").strip()
            for county in counties
            if str(county).strip()
        ]
        if not wanted:
            return pd.DataFrame(columns=self.companies_df.columns)
        rows = self._query_df(
            f"""
SELECT {_COMPANY_COLUMNS}
FROM parent_chunks c
JOIN georgia_counties county
  ON ST_Covers(county.geom, c.geom)
WHERE c.geo IS NOT NULL
  AND lower(county.county_name) = ANY(%s)
ORDER BY county.county_name, c.company, c.updated_location;
""",
            [[county.lower() for county in wanted]],
        )
        return self._limit_to_candidates(rows, candidates)

    def companies_within_radius(
        self,
        lat: float,
        lon: float,
        radius_km: float,
        candidates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        rows = self._query_df(
            f"""
WITH center AS (
    SELECT ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography AS geo
)
SELECT {_COMPANY_COLUMNS},
       ST_Distance(c.geo, center.geo) / {_METERS_PER_KM} AS distance_km,
       ST_Distance(c.geo, center.geo) / {_METERS_PER_KM} * {_KM_TO_MILES} AS distance_miles
FROM parent_chunks c
JOIN center ON TRUE
{_COUNTY_LATERAL_JOIN}
WHERE c.geo IS NOT NULL
  AND ST_DWithin(c.geo, center.geo, %s)
ORDER BY distance_km, c.company;
""",
            [float(lon), float(lat), float(radius_km) * _METERS_PER_KM],
        )
        return self._limit_to_candidates(rows, candidates)

    def companies_near_city(
        self,
        city_name: str,
        radius_km: float = 50.0,
        candidates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        coordinates = self.resolve_place_coordinates(city_name)
        if coordinates is None:
            return pd.DataFrame(columns=[*self.companies_df.columns, "distance_km", "distance_miles"])
        return self.companies_within_radius(
            coordinates[0],
            coordinates[1],
            radius_km,
            candidates,
        )

    @staticmethod
    def _eligible_clause(
        capability_term: Optional[str],
        category_term: Optional[str],
    ) -> tuple[str, list[Any]]:
        clauses = ["geo IS NOT NULL"]
        params: list[Any] = []
        if category_term:
            clauses.append("category ILIKE %s")
            params.append(f"%{category_term}%")
        if capability_term:
            clauses.append(
                "concat_ws(' ', industry_group, product_service, ev_supply_chain_role) ILIKE %s"
            )
            params.append(f"%{capability_term}%")
        return " AND ".join(clauses), params

    def supply_gap_report(
        self,
        capability_term: Optional[str] = None,
        category_term: Optional[str] = None,
        county_scope: Optional[Sequence[str]] = None,
        max_gap_counties: int = 12,
    ) -> dict[str, object]:
        eligible_where, eligible_params = self._eligible_clause(capability_term, category_term)
        scope = [
            str(county).strip().removesuffix(" County").strip().lower()
            for county in (county_scope or [])
            if str(county).strip()
        ]
        scope_clause = "AND lower(g.county_name) = ANY(%s)" if scope else ""
        scope_params: list[Any] = [scope] if scope else []

        gaps = self._query_df(
            f"""
WITH eligible AS (
    SELECT * FROM parent_chunks WHERE {eligible_where}
), covered AS (
    SELECT DISTINCT g.county_name
    FROM georgia_counties g
    JOIN eligible e ON ST_Covers(g.geom, e.geom)
)
SELECT g.county_name AS county,
       nearest.distance_meters / {_METERS_PER_KM} AS nearest_supplier_distance_km,
       nearest.distance_meters / {_METERS_PER_KM} * {_KM_TO_MILES} AS nearest_supplier_distance_miles,
       ST_Y(g.center_geom) AS latitude,
       ST_X(g.center_geom) AS longitude
FROM georgia_counties g
LEFT JOIN covered ON covered.county_name = g.county_name
LEFT JOIN LATERAL (
    SELECT ST_Distance(g.center_geo, e.geo) AS distance_meters
    FROM eligible e
    ORDER BY g.center_geo <-> e.geo
    LIMIT 1
) nearest ON TRUE
WHERE covered.county_name IS NULL
  {scope_clause}
ORDER BY nearest_supplier_distance_km DESC NULLS LAST, g.county_name;
""",
            [*eligible_params, *scope_params],
        )
        coverage = self._query_df(
            f"""
WITH eligible AS (
    SELECT * FROM parent_chunks WHERE {eligible_where}
)
SELECT g.county_name AS county, COUNT(*) AS facility_count
FROM georgia_counties g
JOIN eligible e ON ST_Covers(g.geom, e.geom)
WHERE TRUE {scope_clause}
GROUP BY g.county_name
ORDER BY facility_count DESC, g.county_name;
""",
            [*eligible_params, *scope_params],
        )
        covered_counties = (
            coverage["county"].dropna().astype(str).tolist()
            if "county" in coverage
            else []
        )
        scope_count = len(scope) if scope else len(self.county_names)
        return {
            "covered_counties": sorted(covered_counties),
            "covered_county_count": len(covered_counties),
            "gap_counties": gaps.head(int(max_gap_counties)).to_dict(orient="records"),
            "gap_county_count": int(len(gaps)),
            "scope_county_count": scope_count,
            "coverage_by_county": coverage.head(25).to_dict(orient="records"),
            "spatial_backend": "PostGIS",
        }

    def list_company_names(self) -> list[str]:
        if self.companies_df.empty or "company" not in self.companies_df:
            return []
        return sorted(
            {
                str(name).strip()
                for name in self.companies_df["company"].dropna().astype(str)
                if str(name).strip()
            }
        )
