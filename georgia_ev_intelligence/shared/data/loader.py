import re
import sys
import os
from pathlib import Path

import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE.parent
PACKAGE_DIR = CURRENT_FILE.parents[2]
PROJECT_ROOT = CURRENT_FILE.parents[3]


# -------------------------------------------------------
# Find KB Excel without importing project config
# -------------------------------------------------------
def find_kb_excel() -> Path:
    """
    Priority:
    1. Path passed in command line
    2. GNEM_EXCEL environment variable
    3. Search project folders
    """

    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        if not arg.lower().endswith((".xls", ".xlsx")):
            continue
        path = Path(arg).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Excel file not found: {path}")

    env_path = os.getenv("GNEM_EXCEL")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"GNEM_EXCEL path not found: {path}")

    candidates = []

    for folder in [DATA_DIR, PACKAGE_DIR, PROJECT_ROOT]:
        candidates.extend(folder.rglob("*.xlsx"))

    valid = [
        p for p in candidates
        if not p.name.startswith("~$")
        and "updated_df_debug" not in p.name.lower()
        and "debug" not in p.name.lower()
    ]

    preferred = [
        p for p in valid
        if "gnem" in p.name.lower()
        or "auto" in p.name.lower()
        or "final" in p.name.lower()
    ]

    if preferred:
        return preferred[0]

    if valid:
        return valid[0]

    raise FileNotFoundError(
        "No KB Excel file found. Put the KB .xlsx inside the project folder "
        "or run: python loader.py /full/path/to/GNEM_final_data.xlsx"
    )


# -------------------------------------------------------
# Main loader
# -------------------------------------------------------
def load() -> pd.DataFrame:
    kb_path = find_kb_excel()
    print(f"Using KB file: {kb_path}")

    df = pd.read_excel(kb_path)

    # Normalize column headings only
    df.columns = [_norm_column(c) for c in df.columns]

    if KBColumns.COMPANY not in df.columns:
        raise ValueError(f"'company' column not found. Columns: {df.columns.tolist()}")

    # Keep existing behavior: remove rows without company identity
    df = df.dropna(subset=[KBColumns.COMPANY]).reset_index(drop=True)

    # Normalize values
    df = normalize_dataframe(df)

    # Add row id
    df[KBColumns.ROW_ID] = df.index


    # Final missing handling
    df = df.fillna("Unknown")
    df = df.replace("", "Unknown")

    return df


# -------------------------------------------------------
# Column heading normalization
# Product / Service -> product_service
# EV / Battery Relevant -> ev_battery_relevant
# -------------------------------------------------------
def _norm_column(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


class KBColumns:
    """Column names after loader heading normalization."""

    COMPANY = _norm_column("Company")
    CATEGORY = _norm_column("Category")
    INDUSTRY_GROUP = _norm_column("Industry Group")
    UPDATED_LOCATION = _norm_column("Updated Location")
    ADDRESS = _norm_column("Address")
    LATITUDE = _norm_column("Latitude")
    LONGITUDE = _norm_column("Longitude")
    PRIMARY_FACILITY_TYPE = _norm_column("Primary Facility Type")
    EV_SUPPLY_CHAIN_ROLE = _norm_column("EV Supply Chain Role")
    PRIMARY_OEMS = _norm_column("Primary OEMs")
    SUPPLIER_OR_AFFILIATION_TYPE = _norm_column("Supplier or Affiliation Type")
    EMPLOYMENT = _norm_column("Employment")
    PRODUCT_SERVICE = _norm_column("Product / Service")
    EV_BATTERY_RELEVANT = _norm_column("EV / Battery Relevant")
    CLASSIFICATION_METHOD = _norm_column("Classification Method")

    COMPANY_CLEAN = _norm_column("Company Clean")
    COUNTY = _norm_column("County")
    TIER_CATEGORY_HEURISTIC = _norm_column("Tier Category Heuristic")
    TIER_LEVEL = _norm_column("Tier Level")
    TIER_CONFIDENCE = _norm_column("Tier Confidence")
    OEM_GA = _norm_column("OEM GA")
    INDUSTRY_CODE = _norm_column("Industry Code")
    INDUSTRY_NAME = _norm_column("Industry Name")
    PDF_PAGE = _norm_column("PDF Page")
    IS_ANNOUNCEMENT = _norm_column("Is Announcement")

    ROW_ID = "_row_id"


# -------------------------------------------------------
# Basic text cleaning
# -------------------------------------------------------
def clean_text(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value)

    # Remove line breaks
    value = value.replace("\n", " ").replace("\r", " ")

    # Trim and collapse repeated spaces
    value = re.sub(r"\s+", " ", value).strip()

    if value == "" or value.lower() in {"nan", "none", "null", "na", "n/a"}:
        return "Unknown"

    return value


# -------------------------------------------------------
# Numeric cleaning
# employment, latitude, longitude
# -------------------------------------------------------
def clean_numeric(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip()

    if value == "" or value.lower() in {"nan", "none", "null", "na", "n/a"}:
        return "Unknown"

    # Remove commas: 1,200 -> 1200
    value = value.replace(",", "")

    # Extract number
    match = re.search(r"-?\d+(\.\d+)?", value)

    if not match:
        return "Unknown"

    number = pd.to_numeric(match.group(0), errors="coerce")

    if pd.isna(number):
        return "Unknown"

    return number


# -------------------------------------------------------
# EV / Battery Relevant normalization
# Allowed values:
# Yes, No, Indirect, Unknown
# -------------------------------------------------------
def normalize_ev_battery(value):
    value = clean_text(value)
    lower = value.lower()

    if lower in {"yes", "y", "true", "1"}:
        return "Yes"

    if lower in {"no", "n", "false", "0"}:
        return "No"

    if lower in {"indirect", "indirectly", "partial", "partially"}:
        return "Indirect"

    return "Unknown"


# -------------------------------------------------------
# Supplier / Affiliation Type normalization
# Allowed values:
# Original Equipment Manufacturer
# Automotive supply chain participant
# Unknown
# -------------------------------------------------------
def normalize_supplier_affiliation(value):
    value = clean_text(value)
    lower = value.lower()

    if lower == "unknown":
        return "Unknown"

    if (
        "original equipment manufacturer" in lower
        or lower == "oem"
        or "automaker" in lower
        or "vehicle manufacturer" in lower
    ):
        return "Original Equipment Manufacturer"

    if (
        "automotive supply chain participant" in lower
        or "supplier" in lower
        or "supply chain" in lower
        or "participant" in lower
    ):
        return "Automotive supply chain participant"

    return "Unknown"


# -------------------------------------------------------
# Primary OEMs cleaning
# Clean spaces and standardize separators only
# -------------------------------------------------------
def clean_primary_oems(value):
    value = clean_text(value)

    if value == "Unknown":
        return value

    # Standardize separators
    value = re.sub(r"\s*/\s*", "; ", value)
    value = re.sub(r"\s*,\s*", "; ", value)
    value = re.sub(r"\s*\|\s*", "; ", value)
    value = re.sub(r"\s*&\s*", "; ", value)

    # Remove repeated separators
    parts = [p.strip() for p in value.split(";") if p.strip()]
    value = "; ".join(parts)

    return value if value else "Unknown"


# -------------------------------------------------------
# Product / Service cleaning
# Light cleaning only
# -------------------------------------------------------
def clean_product_service(value):
    return clean_text(value)


# -------------------------------------------------------
# Normalize full dataframe
# -------------------------------------------------------
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:

        # Numeric columns
        if col in {KBColumns.EMPLOYMENT, KBColumns.LATITUDE, KBColumns.LONGITUDE}:
            df[col] = df[col].apply(clean_numeric)

        # Company:
        # Do not change company names.
        # Only remove extra spaces and line breaks.
        elif col == KBColumns.COMPANY:
            df[col] = df[col].apply(clean_text)

        # Product / Service:
        # Preserve original meaning. Only light cleaning.
        elif col == KBColumns.PRODUCT_SERVICE:
            df[col] = df[col].apply(clean_product_service)

        # Primary OEMs:
        # Clean spaces and separators only.
        elif col == KBColumns.PRIMARY_OEMS:
            df[col] = df[col].apply(clean_primary_oems)

        # Supplier / Affiliation Type:
        # Strong normalization to 3 allowed values.
        elif col == KBColumns.SUPPLIER_OR_AFFILIATION_TYPE:
            df[col] = df[col].apply(normalize_supplier_affiliation)

        # EV / Battery Relevant:
        # Strong normalization to 4 allowed values.
        elif col == KBColumns.EV_BATTERY_RELEVANT:
            df[col] = df[col].apply(normalize_ev_battery)

        # All other columns:
        # category, industry_group, location, updated_location,
        # address, primary_facility_type, ev_supply_chain_role,
        # classification_method, etc.
        # Basic cleaning only. No value renaming.
        else:
            df[col] = df[col].apply(clean_text)

    return df



# -------------------------------------------------------
# Debug report
# -------------------------------------------------------
def build_debug_report(df: pd.DataFrame) -> dict:
    sheets = {}

    sheets["updated_df"] = df

    missing_summary = pd.DataFrame({
        "column": df.columns,
        "unknown_count": [(df[c] == "Unknown").sum() for c in df.columns],
        "unknown_percent": [
            round(((df[c] == "Unknown").sum() / len(df)) * 100, 2)
            for c in df.columns
        ],
    })

    sheets["missing_summary"] = missing_summary

    if KBColumns.SUPPLIER_OR_AFFILIATION_TYPE in df.columns:
        supplier_counts = (
            df[KBColumns.SUPPLIER_OR_AFFILIATION_TYPE]
            .value_counts(dropna=False)
            .reset_index()
        )
        supplier_counts.columns = [KBColumns.SUPPLIER_OR_AFFILIATION_TYPE, "count"]
        sheets["supplier_type_values"] = supplier_counts

    if KBColumns.EV_BATTERY_RELEVANT in df.columns:
        ev_counts = (
            df[KBColumns.EV_BATTERY_RELEVANT]
            .value_counts(dropna=False)
            .reset_index()
        )
        ev_counts.columns = [KBColumns.EV_BATTERY_RELEVANT, "count"]
        sheets["ev_battery_values"] = ev_counts

    return sheets


# -------------------------------------------------------
# Run directly
# -------------------------------------------------------
if __name__ == "__main__":
    df = load()

    output_path = DATA_DIR / "updated_df_debug.xlsx"

    sheets = build_debug_report(df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("\nUpdated DataFrame saved successfully.")
    print(f"Output file: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing summary:")
    print(sheets["missing_summary"].to_string(index=False))

    if "supplier_type_values" in sheets:
        print("\nSupplier/Affiliation Type values:")
        print(sheets["supplier_type_values"].to_string(index=False))

    if "ev_battery_values" in sheets:
        print("\nEV/Battery Relevant values:")
        print(sheets["ev_battery_values"].to_string(index=False))
