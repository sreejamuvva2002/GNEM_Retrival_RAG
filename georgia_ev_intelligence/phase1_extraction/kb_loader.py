"""
Phase 1 — KB Loader
Reads GNEM Excel → 205 company rows → stores in PostgreSQL gev_companies table.
Adopted from ev_data_LLM_comparsions/src/kb_loader.py + Kb_Enrichment column mapping.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from shared.config import Config
from shared.db import Company, create_tables, get_session
from shared.logger import get_logger

logger = get_logger("phase1.kb_loader")

# ── Employment overrides: loaded from CSV, not hardcoded ──────────────────────
# File: kb/employment_overrides.csv
# Columns: Company Name, Employment Override, Source, Note
# Edit the CSV to add/remove overrides — no Python changes needed.
def _load_employment_overrides() -> dict[str, float]:
    """
    Load employment overrides from kb/employment_overrides.csv.
    Returns dict: company_name → correct_georgia_employment.
    Falls back to empty dict if file not found (no overrides applied).
    """
    csv_path = Path(__file__).resolve().parents[2] / "kb" / "employment_overrides.csv"
    if not csv_path.exists():
        logger.warning("employment_overrides.csv not found at %s — no caps applied", csv_path)
        return {}
    try:
        df = pd.read_csv(csv_path)
        overrides = {}
        for _, row in df.iterrows():
            name = str(row.get("Company Name", "")).strip()
            val  = row.get("Employment Override")
            if name and val is not None:
                try:
                    overrides[name] = float(val)
                except (ValueError, TypeError):
                    pass
        logger.info("Loaded %d employment overrides from CSV", len(overrides))
        return overrides
    except Exception as exc:
        logger.warning("Failed to load employment_overrides.csv: %s", exc)
        return {}

# Column name mapping from GNEM Excel → our DB fields
# Tolerant: strips whitespace, lowercases for matching
# NOTE: verified against actual GNEM Excel headers:
#   'company', 'category', 'industry group', 'updated location',
#   'address', 'latitude', 'longitude', 'primary facility type',
#   'ev supply chain role', 'primary oems', 'supplier or affiliation type',
#   'employment', 'product / service', 'ev / battery relevant', 'classification method'
_COL_MAP = {
    "company": "company_name",
    "category": "tier",
    "ev supply chain role": "ev_supply_chain_role",
    "primary oems": "primary_oems",
    "ev / battery relevant": "ev_battery_relevant",
    "industry group": "industry_group",
    "primary facility type": "facility_type",    # Manufacturing Plant / R&D etc.
    "employment": "employment",
    "product / service": "products_services",
    "products / services": "products_services",
    "classification method": "classification_method",
    "supplier or affiliation type": "supplier_affiliation_type",
    "latitude": "latitude",
    "longitude": "longitude",
}


def _find_gnem_excel() -> Path:
    """
    Locate the GNEM Excel file.
    Prefers the cleaned version (GNEM_Cleaned.xlsx) if it exists.
    Falls back to the original if cleaned is not found.
    Run scripts/clean_excel.py first to create the cleaned version.
    """
    kb_dir = Path(__file__).resolve().parents[2] / "kb"
    candidates = [
        kb_dir / "GNEM_Cleaned.xlsx",                                  # ← cleaned first
        kb_dir / "GNEM - Auto Landscape Lat Long Updated.xlsx",         # ← original fallback
        Path(__file__).resolve().parents[2] / "ev_data_LLM_comparsions" / "GNEM - Auto Landscape Lat Long Updated.xlsx",
        Path(__file__).resolve().parents[2] / "GNEM - Auto Landscape Lat Long Updated.xlsx",
    ]
    for path in candidates:
        if path.exists():
            logger.info("Found GNEM Excel at: %s", path)
            return path
    raise FileNotFoundError(
        "Cannot find GNEM Excel file. Run scripts/clean_excel.py first to create kb/GNEM_Cleaned.xlsx\n"
        f"Searched: {[str(p) for p in candidates]}"
    )



def _parse_employment(raw: Any) -> float | None:
    """Parse employment value — handles '1,500', '1500', '500-1000', None."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).replace(",", "").strip()
    # Handle ranges like "500-1000" → take lower bound
    if "-" in text:
        text = text.split("-")[0].strip()
    text = text.replace("+", "").replace("~", "").strip()
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase stripped for flexible matching."""
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def load_gnem_dataframe() -> pd.DataFrame:
    """Load and normalize the GNEM Excel. Returns 205-row DataFrame."""
    path = _find_gnem_excel()
    df = pd.read_excel(path, engine="openpyxl")
    df = _normalize_columns(df)
    logger.info("Loaded GNEM Excel: %d rows × %d columns", len(df), len(df.columns))
    return df


def _parse_float(raw: Any) -> float | None:
    """Safely parse any value to float, return None if not parseable."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        return float(str(raw).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _detect_shifted_row(row: pd.Series) -> bool:
    """
    Detect if a row has its columns shifted (data quality issue in GNEM Excel).
    Pattern: longitude column contains text instead of a numeric value.
    """
    lon = row.get("longitude")
    if lon is None:
        return False
    if isinstance(lon, float) and pd.isna(lon):
        return False
    if isinstance(lon, str) and not lon.replace(".", "").replace("-", "").strip().isdigit():
        return True
    return False


def _row_to_company_dict(row: pd.Series) -> dict[str, Any]:
    """Convert a GNEM Excel row to a dict matching Company model fields."""
    result: dict[str, Any] = {}
    for excel_col, db_field in _COL_MAP.items():
        if excel_col in row.index:
            val = row[excel_col]
            is_null = pd.isna(val) if isinstance(val, float) else (val is None)
            if is_null:
                result[db_field] = None
            else:
                result[db_field] = str(val).strip() if isinstance(val, str) else val

    # Explicitly cast float-only columns to prevent DataError
    result["latitude"] = _parse_float(row.get("latitude"))
    result["longitude"] = _parse_float(row.get("longitude"))
    result["employment"] = _parse_employment(row.get("employment"))

    # Parse 'updated location' column: e.g. 'Savannah, Chatham County'
    # Split on first comma: city, county
    loc_raw = row.get("updated location") or row.get("location")
    if loc_raw and not pd.isna(loc_raw) if isinstance(loc_raw, float) else loc_raw:
        loc_str = str(loc_raw).strip()
        if "," in loc_str:
            parts = [p.strip() for p in loc_str.split(",", 1)]
            result["location_city"] = result.get("location_city") or parts[0]
            result["location_county"] = result.get("location_county") or parts[1]
        else:
            result["location_city"] = result.get("location_city") or loc_str
    result["location_state"] = "Georgia"

    # Ensure company_name is set
    if not result.get("company_name"):
        for alt in ("company name", "name", "organization"):
            if alt in row.index and not pd.isna(row[alt]):
                result["company_name"] = str(row[alt]).strip()
                break
    return result


def load_companies_from_excel() -> list[dict[str, Any]]:
    """
    Load all 205 companies from GNEM Excel as list of dicts.
    Handles known data quality issues:
      - Row 204 (ZF Gainesville LLC): columns shifted by 1-2 positions
        → detected by non-numeric longitude → skipped with warning
    """
    df = load_gnem_dataframe()
    companies = []
    skipped_dq = 0
    for _, row in df.iterrows():
        # Skip rows with known data quality issues (shifted columns)
        if _detect_shifted_row(row):
            name = row.get("company", "unknown")
            logger.warning(
                "Skipping '%s' — column-shifted row (data quality issue in GNEM Excel). "
                "Longitude contains text: %r",
                name, row.get("longitude")
            )
            skipped_dq += 1
            continue

        company = _row_to_company_dict(row)
        if not company.get("company_name"):
            logger.warning("Skipping row with no company name: %s", dict(row))
            continue

        # Enforce max string lengths matching DB VARCHAR constraints
        _STR_MAX = {
            "tier": 50,
            "ev_supply_chain_role": 200,
            "primary_oems": 200,
            "ev_battery_relevant": 100,
            "industry_group": 200,
            "location_city": 200,
            "location_county": 200,
            "classification_method": 100,
            "supplier_affiliation_type": 200,
        }
        for field, max_len in _STR_MAX.items():
            val = company.get(field)
            if isinstance(val, str) and len(val) > max_len:
                company[field] = val[:max_len]

        companies.append(company)

    if skipped_dq:
        logger.warning("Skipped %d rows due to data quality issues (shifted columns in GNEM Excel)", skipped_dq)
    logger.info("Parsed %d companies from GNEM Excel (%d skipped for data quality)", len(companies), skipped_dq)

    # Apply employment overrides from CSV (not hardcoded)
    overrides = _load_employment_overrides()
    capped = 0
    for company in companies:
        name = company.get("company_name", "")
        if name in overrides:
            original = company.get("employment")
            company["employment"] = overrides[name]
            logger.info("[OVERRIDE] %s: %s → %.0f (from CSV)", name, original, overrides[name])
            capped += 1
    if capped:
        logger.info("Applied employment overrides to %d companies", capped)

    return companies


def sync_companies_to_db(companies: list[dict[str, Any]]) -> tuple[int, int]:
    """
    Upsert companies into gev_companies table.
    Skips rows that fail validation (data quality issues in source Excel).
    Returns (inserted, updated) counts.
    """
    create_tables()
    session = get_session()
    inserted = updated = skipped = 0

    try:
        for data in companies:
            name = data.get("company_name")
            if not name:
                continue

            # Safety: ensure float columns are actually floats
            for float_col in ("latitude", "longitude", "employment"):
                val = data.get(float_col)
                if val is not None:
                    try:
                        data[float_col] = float(val)
                    except (ValueError, TypeError):
                        logger.warning(
                            "Skipping invalid %s value %r for company %s",
                            float_col, val, name
                        )
                        data[float_col] = None

            # Safety: ensure string columns are strings (prevent int leak from Excel)
            for str_col in (
                "tier", "ev_supply_chain_role", "primary_oems",
                "ev_battery_relevant", "industry_group", "location_city",
                "location_county", "products_services", "classification_method",
                "supplier_affiliation_type",
            ):
                val = data.get(str_col)
                if val is not None and not isinstance(val, str):
                    data[str_col] = str(val)[:200]

            # Only keep fields that exist on the Company model
            safe_data = {k: v for k, v in data.items() if hasattr(Company, k)}

            try:
                existing = session.query(Company).filter_by(company_name=name).first()
                if existing:
                    for field, value in safe_data.items():
                        if field != "company_name":
                            setattr(existing, field, value)
                    updated += 1
                else:
                    company = Company(**safe_data)
                    session.add(company)
                    inserted += 1
                session.flush()  # Catch errors row-by-row
            except Exception as row_exc:
                session.rollback()  # Rollback just this row
                logger.warning("Skipping company %r due to error: %s", name, row_exc)
                skipped += 1
                # Re-open session after rollback
                session = get_session()
                continue

        session.commit()
        logger.info(
            "Companies synced: %d inserted, %d updated, %d skipped (data errors)",
            inserted, updated, skipped
        )
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return inserted, updated


def get_all_companies_from_db() -> list[dict[str, Any]]:
    """Fetch all companies from PostgreSQL as list of dicts (for pipeline use)."""
    session = get_session()
    try:
        rows = session.query(Company).order_by(Company.company_name).all()
        return [
            {
                "id": c.id,
                "company_name": c.company_name,
                "tier": c.tier,
                "ev_supply_chain_role": c.ev_supply_chain_role,
                "primary_oems": c.primary_oems,
                "ev_battery_relevant": c.ev_battery_relevant,
                "industry_group": c.industry_group,
                "location_city": c.location_city,
                "location_county": c.location_county,
                "location_state": c.location_state,
                "employment": c.employment,
                "products_services": c.products_services,
                "latitude": c.latitude,
                "longitude": c.longitude,
            }
            for c in rows
        ]
    finally:
        session.close()


def build_document_text(company: dict[str, Any]) -> str:
    """
    Build a structured text representation of a company for embedding.
    Pattern adopted directly from ev_data_LLM_comparsions/src/kb_loader.py.
    This text is what gets embedded and stored in Qdrant.
    """
    location = " | ".join(filter(None, [
        company.get("location_city"),
        company.get("location_county"),
        company.get("location_state") or "Georgia",
    ]))
    return (
        f"Company: {company.get('company_name', '')} | "
        f"Tier: {company.get('tier', '')} | "
        f"Industry: {company.get('industry_group', '')} | "
        f"Location: {location} | "
        f"EV Role: {company.get('ev_supply_chain_role', '')} | "
        f"OEMs: {company.get('primary_oems', '')} | "
        f"EV / Battery Relevant: {company.get('ev_battery_relevant', '')} | "
        f"Employment: {company.get('employment', '')} | "
        f"Products: {company.get('products_services', '')} | "
        f"Classification: {company.get('classification_method', '')} | "
        f"Affiliation: {company.get('supplier_affiliation_type', '')}"
    )
