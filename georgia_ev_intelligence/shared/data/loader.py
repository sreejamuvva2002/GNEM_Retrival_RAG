import re
from html import unescape
from pathlib import Path

import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
OUTPUTS_DIR = CURRENT_FILE.parents[2] / "outputs"
NORMALIZED_KB_PATH = OUTPUTS_DIR / "Normalized_kb.xlsx"
KB_EXCEL_PATH = Path(
    "/Users/sreejamuvva/Desktop/GNEM_Retrival_RAG/kb/"
    "GNEM - Auto Landscape Lat Long Updated.xlsx"
)


# -------------------------------------------------------
# Find KB Excel without importing project config
# -------------------------------------------------------
def find_kb_excel() -> Path:
    """
    Return the configured KB Excel file.
    """

    if KB_EXCEL_PATH.exists():
        return KB_EXCEL_PATH

    raise FileNotFoundError(f"KB Excel file not found: {KB_EXCEL_PATH}")


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
    COMPANY = _norm_column("Company")
    CATEGORY = _norm_column("Category")
    INDUSTRY_GROUP = _norm_column("Industry Group")
    LOCATION = _norm_column("Location")
    UPDATED_LOCATION = _norm_column("Updated Location")
    ADDRESS = _norm_column("Address")
    PRIMARY_FACILITY_TYPE = _norm_column("Primary Facility Type")
    EV_SUPPLY_CHAIN_ROLE = _norm_column("EV Supply Chain Role")
    PRIMARY_OEMS = _norm_column("Primary OEMs")
    OEM_FOOTPRINT = _norm_column("OEM (Footprint)")
    SUPPLIER_OR_AFFILIATION_TYPE = _norm_column("Supplier or Affiliation Type")
    PRODUCT_SERVICE = _norm_column("Product / Service")
    EV_BATTERY_RELEVANT = _norm_column("EV / Battery Relevant")
    CLASSIFICATION_METHOD = _norm_column("Classification Method")
    LATITUDE = _norm_column("Latitude")
    LONGITUDE = _norm_column("Longitude")
    EMPLOYMENT = _norm_column("Employment")

    ROW_ID = "_row_id"


MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a"}


def _normalize_formatting(value: str) -> str:
    value = unescape(str(value))
    value = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", value)
    value = re.sub(r"[\u00a0\t\r\n]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_STRINGS


def clean_text(value):
    if pd.isna(value):
        return "Unknown"

    value = _normalize_formatting(value)

    if _is_missing(value):
        return "Unknown"

    return value


def clean_missing_only(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value)

    if _is_missing(value):
        return "Unknown"

    return value


def clean_numeric(value):
    if pd.isna(value):
        return "Unknown"

    value = clean_text(value)

    if value == "Unknown":
        return "Unknown"

    value = value.replace(",", "").strip()

    if not re.fullmatch(r"-?\d+(\.\d+)?", value):
        return "Unknown"

    number = pd.to_numeric(value, errors="coerce")

    if pd.isna(number):
        return "Unknown"

    return number


def clean_company(value):
    value = clean_text(value)
    return value if value == "Unknown" else value.lower()


def clean_category(value):
    value = clean_text(value)

    if value == "Unknown":
        return value

    value = re.sub(r"[()]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\boem\b", "OEM", value, flags=re.IGNORECASE)

    if value.casefold() == "oem footprint":
        return "OEM Footprint"

    return value


def clean_oem_footprint(value):
    value = clean_text(value)

    if value == "Unknown":
        return value

    value = re.sub(r"[()]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\boem\b", "OEM", value, flags=re.IGNORECASE)

    return value


def clean_primary_facility_type(value):
    value = clean_text(value)

    if value == "Unknown":
        return value

    value = normalize_separators(value)
    value = re.sub(r"\boem\b", "OEM", value, flags=re.IGNORECASE)
    value = re.sub(r"\br&d\b", "R&D", value, flags=re.IGNORECASE)

    if value.casefold() == "manufacturing plant":
        return "Manufacturing Plant"

    return value


def normalize_separators(value):
    value = clean_text(value)

    if value == "Unknown":
        return value

    value = re.sub(r"\s*([,;/|])\s*", r"\1 ", value)
    value = re.sub(r"\s+", " ", value).strip(" ,;/|")
    return value if value else "Unknown"


def clean_product_service(value):
    value = clean_text(value)

    if value == "Unknown":
        return value

    value = re.sub(r"\s*([;/|])\s*", r" \1 ", value)
    value = re.sub(r"\s+", " ", value).strip(" \t-;,.|/")
    return value if value else "Unknown"


# -------------------------------------------------------
# Normalize full dataframe
# -------------------------------------------------------
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    text_normalizers = {
        KBColumns.COMPANY: clean_company,
        KBColumns.CATEGORY: clean_category,
        KBColumns.UPDATED_LOCATION: clean_missing_only,
        KBColumns.PRIMARY_FACILITY_TYPE: clean_primary_facility_type,
        KBColumns.PRIMARY_OEMS: normalize_separators,
        KBColumns.OEM_FOOTPRINT: clean_oem_footprint,
        KBColumns.PRODUCT_SERVICE: clean_product_service,
    }

    for col in df.columns:

        # Numeric columns
        if col in {KBColumns.EMPLOYMENT, KBColumns.LATITUDE, KBColumns.LONGITUDE}:
            df[col] = df[col].apply(clean_numeric)

        # Column-specific text normalization where requested; fallback to basic cleanup.
        else:
            normalizer = text_normalizers.get(col, clean_text)
            df[col] = df[col].apply(normalizer)

    return df


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

    # Remove only rows without company identity
    df = df.dropna(subset=[KBColumns.COMPANY]).reset_index(drop=True)

    # Normalize values without changing original meaning
    df = normalize_dataframe(df)

    # Add row id for downstream mapping
    df[KBColumns.ROW_ID] = df.index

    # Final missing handling
    df = df.fillna("Unknown")
    df = df.replace("", "Unknown")

    return df


# -------------------------------------------------------
# Debug report
# -------------------------------------------------------
def build_debug_report(df: pd.DataFrame) -> dict:
    sheets = {}

    sheets["Normalized_kb"] = df

    missing_summary = pd.DataFrame({
        "column": df.columns,
        "unknown_count": [(df[c] == "Unknown").sum() for c in df.columns],
        "unknown_percent": [
            round(((df[c] == "Unknown").sum() / len(df)) * 100, 2)
            for c in df.columns
        ],
    })

    sheets["missing_summary"] = missing_summary

    # Value-count sheets for inspection only.
    # These do not change the dataframe.
    for col in [
        KBColumns.CATEGORY,
        KBColumns.PRIMARY_FACILITY_TYPE,
        KBColumns.SUPPLIER_OR_AFFILIATION_TYPE,
        KBColumns.EV_BATTERY_RELEVANT,
        KBColumns.EV_SUPPLY_CHAIN_ROLE,
    ]:
        if col in df.columns:
            value_counts = df[col].value_counts(dropna=False).reset_index()
            value_counts.columns = [col, "count"]
            sheets[f"{col}_values"] = value_counts

    return sheets


# -------------------------------------------------------
# Run directly
# -------------------------------------------------------
if __name__ == "__main__":
    df = load()

    output_path = NORMALIZED_KB_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sheets = build_debug_report(df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print("\nUpdated DataFrame saved successfully.")
    print(f"Output file: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing summary:")
    print(sheets["missing_summary"].to_string(index=False))
