#!/usr/bin/env python3
"""Idempotent PostGIS migration for the route execution stage.

Prepares the database for ``geo_search`` / ``disruption_analysis`` (execution
README §4-6) WITHOUT ``ogr2ogr`` (which is not installed here):

  1. enable the PostGIS extension;
  2. add point columns ``geom`` (geometry) and ``geo`` (geography) to
     ``parent_chunks`` and populate them from latitude/longitude;
  3. build a ``georgia_counties`` table from the bundled Georgia county GeoJSON
     (``streamlit_ui/data/Counties_Georgia.geojson``) using ``ST_GeomFromGeoJSON``,
     plus per-county centre points.

Additive only — it never drops or rewrites existing ``parent_chunks`` data, so
the rest of the pipeline is unaffected. Safe to re-run.

Run from the project root:

    python scripts/setup_postgis.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from georgia_ev_intelligence.route_execution.db import get_connection

DEFAULT_GEOJSON = (
    PROJECT_ROOT
    / "georgia_ev_intelligence"
    / "streamlit_ui"
    / "data"
    / "Counties_Georgia.geojson"
)

_ENABLE_POSTGIS = "CREATE EXTENSION IF NOT EXISTS postgis;"

_PARENT_GEO_SQL = """
ALTER TABLE parent_chunks ADD COLUMN IF NOT EXISTS geom geometry(Point, 4326);
ALTER TABLE parent_chunks ADD COLUMN IF NOT EXISTS geo  geography(Point, 4326);

UPDATE parent_chunks
SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326),
    geo  = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
WHERE latitude IS NOT NULL
  AND longitude IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parent_chunks_geom ON parent_chunks USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_parent_chunks_geo  ON parent_chunks USING GIST (geo);
"""

_CREATE_COUNTIES_SQL = """
CREATE TABLE IF NOT EXISTS georgia_counties (
    county_id    SERIAL PRIMARY KEY,
    county_name  TEXT,
    geom         geometry(MultiPolygon, 4326),
    center_geom  geometry(Point, 4326),
    center_geo   geography(Point, 4326)
);
"""

_INSERT_COUNTY_SQL = """
INSERT INTO georgia_counties (county_name, geom)
VALUES (
    %(name)s,
    ST_Multi(ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON(%(geojson)s), 4326)))
);
"""

_COUNTY_CENTERS_SQL = """
UPDATE georgia_counties
SET center_geom = ST_PointOnSurface(geom),
    center_geo  = ST_PointOnSurface(geom)::geography
WHERE geom IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_georgia_counties_geom    ON georgia_counties USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_georgia_counties_center  ON georgia_counties USING GIST (center_geo);
"""


def _county_name(props: dict) -> str | None:
    name = (
        props.get("NAME10")
        or props.get("NAME")
        or props.get("NAMELSAD10", "").replace("County", "").strip()
    )
    return str(name).strip() if name else None


def _load_counties(conn, geojson_path: Path) -> int:
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = payload.get("features", [])

    with conn.cursor() as cur:
        cur.execute(_CREATE_COUNTIES_SQL)
        # Re-runnable: clear then repopulate from the source GeoJSON.
        cur.execute("DELETE FROM georgia_counties;")

        inserted = 0
        for feature in features:
            name = _county_name(feature.get("properties", {}) or {})
            geometry = feature.get("geometry")
            if not name or not geometry:
                continue
            cur.execute(
                _INSERT_COUNTY_SQL,
                {"name": name, "geojson": json.dumps(geometry)},
            )
            inserted += 1

        cur.execute(_COUNTY_CENTERS_SQL)

    return inserted


def migrate(geojson_path: Path) -> None:
    if not geojson_path.exists():
        raise FileNotFoundError(f"County GeoJSON not found: {geojson_path}")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print("Enabling PostGIS extension…")
            cur.execute(_ENABLE_POSTGIS)
            print("Adding/populating geom/geo on parent_chunks…")
            cur.execute(_PARENT_GEO_SQL)

        print(f"Loading counties from {geojson_path.name}…")
        n_counties = _load_counties(conn, geojson_path)

        conn.commit()
        print(f"Done. Loaded {n_counties} counties.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PostGIS migration for geo routes.")
    parser.add_argument(
        "--geojson",
        type=Path,
        default=DEFAULT_GEOJSON,
        help="Georgia county GeoJSON (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        migrate(args.geojson)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
