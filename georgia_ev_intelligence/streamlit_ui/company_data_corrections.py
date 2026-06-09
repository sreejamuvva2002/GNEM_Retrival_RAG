"""Verified corrections applied over known location errors in the source workbook."""
from __future__ import annotations

import re
from typing import Any, Dict

import pandas as pd


# Hitachi: official Americas directory and OSHA list the Georgia plant at
# 1000 Unisia Drive, Monroe. Honda's official manufacturing site places its
# Georgia transmission plant in Tallapoosa. Coordinates are OSM-geocoded.
COMPANY_DATA_CORRECTIONS: Dict[str, Dict[str, Any]] = {
    "hitachi astemo americas inc": {
        "location": "Monroe, Walton County",
        "address": "1000 Unisia Drive, Monroe, GA 30655",
        "city": "Monroe",
        "county": "Walton",
        "latitude": 33.8086669,
        "longitude": -83.6764364,
        "coordinate_source": "verified_openstreetmap",
    },
    "honda development manufacturing": {
        "location": "Tallapoosa, Haralson County",
        "address": "550 Honda Pkwy, Tallapoosa, GA 30176",
        "city": "Tallapoosa",
        "county": "Haralson",
        "latitude": 33.6932981,
        "longitude": -85.2728142,
        "coordinate_source": "verified_openstreetmap",
    },
}


def apply_company_data_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with verified company-location corrections applied."""
    if df.empty or "company" not in df.columns:
        return df.copy()

    out = df.copy()
    normalized = out["company"].fillna("").astype(str).map(_normalize_company)
    for company_key, values in COMPANY_DATA_CORRECTIONS.items():
        mask = normalized == company_key
        if not mask.any():
            continue
        for column, value in values.items():
            if column not in out.columns:
                out[column] = None
            out.loc[mask, column] = value
    return out


def _normalize_company(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
