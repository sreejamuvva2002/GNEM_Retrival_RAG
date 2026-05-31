"""One-shot ingestion: kb GNEM xlsx → streamlit_ui/data/gnem.duckdb companies table.

Adapted from PAST_GEO_MAP_VIEW/backend/ingestion.py, scoped down to the company-table
build only. We skip FAISS / chunk indexing because the new project uses the existing
hybrid_retrieval pipeline (BM25 + pgvector on Neon) for the RAG layer; this duckdb
is consumed only by the spatial engine that drives the map.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd

from ..spatial.county_geo import (
    infer_county_from_point,
    load_county_centroids,
    load_county_geometries,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXCEL_PATH = PROJECT_ROOT / "kb" / "GNEM - Auto Landscape Lat Long Updated.xlsx"
PACKAGE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_GEOJSON_PATH = PACKAGE_DATA_DIR / "Counties_Georgia.geojson"
DEFAULT_DB_PATH = PACKAGE_DATA_DIR / "gnem.duckdb"

CITY_COUNTY_FALLBACK = {
    "atlanta": "Fulton",
    "alpharetta": "Fulton",
    "augusta": "Richmond",
    "bainbridge": "Decatur",
    "columbus": "Muscogee",
    "macon": "Bibb",
    "marietta": "Cobb",
    "savannah": "Chatham",
    "statesboro": "Bulloch",
    "west point": "Troup",
}


def _clean_column_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return cleaned.strip("_")


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_column_name(col) for col in df.columns]
    return df


def _normalize_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _safe_float(value: object) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_city_county(location: object) -> Tuple[Optional[str], Optional[str]]:
    if pd.isna(location):
        return None, None
    text = str(location).strip()
    if not text:
        return None, None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    city = parts[0].title() if parts else None
    county_match = re.search(r"([A-Za-z][A-Za-z\s\-']+?)\s+County", text, flags=re.IGNORECASE)
    county = county_match.group(1).strip().title() if county_match else None
    if city and (city.lower().endswith(" county") or city.lower() in {"georgia", "ga"}):
        city = None
    if not county and len(parts) > 1:
        maybe_county = parts[1].replace("County", "").strip()
        county = maybe_county.title() if maybe_county else None
    return city, county


def _extract_city_from_address(address: object) -> Optional[str]:
    if pd.isna(address):
        return None
    text = str(address).strip()
    if not text:
        return None
    match = re.search(
        r"(?:^|,\s*)([A-Za-z][A-Za-z\s\-']+?),\s*(?:GA|Georgia)\b",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _attach_coordinates(
    df: pd.DataFrame,
    county_centroids: Dict[str, Tuple[float, float]],
    county_geometries: Sequence[dict],
) -> pd.DataFrame:
    out = df.copy()
    cities: List[Optional[str]] = []
    counties: List[Optional[str]] = []
    lats: List[Optional[float]] = []
    lons: List[Optional[float]] = []
    sources: List[str] = []

    out["latitude"] = (
        pd.to_numeric(out["latitude"], errors="coerce") if "latitude" in out.columns else np.nan
    )
    out["longitude"] = (
        pd.to_numeric(out["longitude"], errors="coerce") if "longitude" in out.columns else np.nan
    )
    if "address" not in out.columns:
        out["address"] = ""

    for _, row in out.iterrows():
        location_text = _normalize_cell(row.get("location") or row.get("updated_location"))
        address_text = _normalize_cell(row.get("address"))
        city, county = _extract_city_county(location_text)

        if not city:
            city = _extract_city_from_address(address_text)
        if not county and city:
            county = CITY_COUNTY_FALLBACK.get(city.strip().lower())

        lat = _safe_float(row.get("latitude"))
        lon = _safe_float(row.get("longitude"))
        source = "source_excel" if (lat is not None and lon is not None) else "missing"

        if county_geometries and lat is not None and lon is not None:
            inferred = infer_county_from_point(lat=lat, lon=lon, county_geometries=county_geometries)
            if inferred:
                county = inferred

        if (lat is None or lon is None) and county:
            key = county.strip().lower()
            if key in county_centroids:
                lat, lon = county_centroids[key]
                source = "county_centroid"

        if lat is None or lon is None:
            source = "missing"

        cities.append(city)
        counties.append(county)
        lats.append(lat)
        lons.append(lon)
        sources.append(source)

    out["city"] = cities
    out["county"] = counties
    out["latitude"] = lats
    out["longitude"] = lons
    out["coordinate_source"] = sources
    return out


COMPANIES_TABLE_COLUMNS = [
    "company",
    "category",
    "industry_group",
    "location",
    "address",
    "city",
    "county",
    "ev_supply_chain_role",
    "primary_oems",
    "supplier_or_affiliation_type",
    "employment",
    "product_service",
    "ev_battery_relevant",
    "primary_facility_type",
    "latitude",
    "longitude",
    "coordinate_source",
]


def _prepare_companies_dataframe(excel_path: Path, geojson_path: Path) -> pd.DataFrame:
    df = _clean_columns(pd.read_excel(excel_path, sheet_name=0))

    # Normalize source-column names to the schema spatial_engine expects.
    if "updated_location" in df.columns and "location" not in df.columns:
        df = df.rename(columns={"updated_location": "location"})
    if "product_service" not in df.columns and "product___service" in df.columns:
        df = df.rename(columns={"product___service": "product_service"})
    if "ev_battery_relevant" not in df.columns and "ev___battery_relevant" in df.columns:
        df = df.rename(columns={"ev___battery_relevant": "ev_battery_relevant"})

    if "employment" in df.columns:
        df["employment"] = (
            df["employment"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"nan": None, "None": None, "": None})
        )
        df["employment"] = pd.to_numeric(df["employment"], errors="coerce")

    county_centroids = load_county_centroids(geojson_path)
    county_geometries = load_county_geometries(geojson_path)
    df = _attach_coordinates(df, county_centroids=county_centroids, county_geometries=county_geometries)

    for col in COMPANIES_TABLE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[COMPANIES_TABLE_COLUMNS].drop_duplicates(keep="first").reset_index(drop=True)


def _write_companies(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.register("df_companies", df)
        con.execute("CREATE OR REPLACE TABLE companies AS SELECT * FROM df_companies")


def build_companies_db(
    excel_path: Path = DEFAULT_EXCEL_PATH,
    geojson_path: Path = DEFAULT_GEOJSON_PATH,
    db_path: Path = DEFAULT_DB_PATH,
) -> Path:
    """Build the companies DuckDB from the GNEM xlsx. Returns the db path."""
    if not excel_path.exists():
        raise FileNotFoundError(f"GNEM workbook missing: {excel_path}")
    if not geojson_path.exists():
        raise FileNotFoundError(f"Counties_Georgia.geojson missing: {geojson_path}")

    df = _prepare_companies_dataframe(excel_path=excel_path, geojson_path=geojson_path)
    _write_companies(df, db_path)
    return db_path


def ensure_companies_db(
    excel_path: Path = DEFAULT_EXCEL_PATH,
    geojson_path: Path = DEFAULT_GEOJSON_PATH,
    db_path: Path = DEFAULT_DB_PATH,
) -> Path:
    """Build the db only if it does not already exist."""
    if db_path.exists():
        return db_path
    return build_companies_db(excel_path=excel_path, geojson_path=geojson_path, db_path=db_path)


if __name__ == "__main__":
    out = build_companies_db()
    print(f"Wrote companies table → {out}")
