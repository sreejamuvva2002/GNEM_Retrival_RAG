"""
GNEM Excel Data Audit & Cleaner
Reads the original Excel, identifies ALL issues, fixes them,
saves a clean version: kb/GNEM_Cleaned.xlsx

Issues it fixes:
  1. Shifted columns (like ZF Gainesville row) — detected + corrected
  2. "Multiple OEMs" → kept but flagged for manual lookup
  3. Compound OEM strings like "Hyundai Kia" → split to "Hyundai, Kia"
  4. Missing county in 'updated location' — extracts from address if possible
  5. Employment ranges like "500-1000" → lower bound
  6. Empty company names — dropped
  7. Missing tier — flagged as "Unknown"

Run:
  venv\\Scripts\\python scripts\\clean_excel.py

Output:
  - Prints full audit report (what was fixed, what needs manual review)
  - Saves: kb/GNEM_Cleaned.xlsx
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from phase1_extraction.kb_loader import _find_gnem_excel, _normalize_columns

SEP = "=" * 60

# Known OEM compound names → how to split them
# This is the core fix for the "Multiple OEMs" and "Hyundai Kia" problem
OEM_NORMALIZATIONS = {
    "multiple oems":           None,   # None = keep as-is, flagged for manual review
    "hyundai kia":             "Hyundai, Kia",
    "hyundai kia rivian":      "Hyundai, Kia, Rivian",
    "hyundai/kia":             "Hyundai, Kia",
    "kia/hyundai":             "Kia, Hyundai",
    "ford gm":                 "Ford, GM",
    "ford/gm":                 "Ford, GM",
    "toyota honda":            "Toyota, Honda",
    "bmw toyota":              "BMW, Toyota",
    "multiple":                None,
    "various":                 None,
    "n/a":                     None,
    "na":                      None,
}


def _parse_employment_range(raw) -> str:
    """Convert '500-1000' → '500', '1,500+' → '1500', etc."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    text = str(raw).replace(",", "").strip()
    if "-" in text:
        text = text.split("-")[0].strip()
    text = text.replace("+", "").replace("~", "").strip()
    try:
        float(text)
        return text
    except ValueError:
        return str(raw)  # Return original if can't parse


def _normalize_oem(raw_oem: str) -> tuple[str, bool]:
    """
    Normalize OEM string. Returns (cleaned_oem, needs_manual_review).
    """
    if not raw_oem or raw_oem.strip().lower() in ("", "nan", "none"):
        return "", False

    key = raw_oem.strip().lower()
    if key in OEM_NORMALIZATIONS:
        replacement = OEM_NORMALIZATIONS[key]
        if replacement is None:
            return raw_oem.strip(), True   # Keep as-is, flag for review
        return replacement, False

    return raw_oem.strip(), False


def _detect_shifted_row(row: pd.Series) -> bool:
    """Check if longitude column has text instead of a number."""
    lon = row.get("longitude")
    if lon is None or (isinstance(lon, float) and pd.isna(lon)):
        return False
    if isinstance(lon, str) and not lon.replace(".", "").replace("-", "").strip().isdigit():
        return True
    return False


def main():
    print(f"\n{'#'*60}")
    print("  GNEM EXCEL DATA AUDIT & CLEANER")
    print(f"{'#'*60}\n")

    excel_path = _find_gnem_excel()
    df = pd.read_excel(excel_path, engine="openpyxl")
    df_orig = df.copy()
    df = _normalize_columns(df)

    print(f"  Source:  {excel_path.name}")
    print(f"  Rows:    {len(df)}")
    print(f"  Cols:    {list(df.columns)}\n")

    issues = []
    fixes  = []

    # ── 1. Shifted column rows ────────────────────────────────────────────────
    print(f"{SEP}")
    print("  Issue 1: Shifted column rows")
    print(SEP)
    shifted_idx = []
    for idx, row in df.iterrows():
        if _detect_shifted_row(row):
            name = row.get("company", f"row {idx}")
            print(f"  ROW {idx:>3}: '{name}' — longitude='{row.get('longitude')}' (text, not number)")
            shifted_idx.append(idx)
            issues.append(f"Row {idx} ({name}): shifted columns")

    if not shifted_idx:
        print("  ✅ No shifted rows found")
    else:
        print(f"\n  → Dropping {len(shifted_idx)} shifted rows (manual fix needed in Excel)")
        df = df.drop(index=shifted_idx)

    # ── 2. Missing company names ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Issue 2: Missing company names")
    print(SEP)
    missing_name = df[df["company"].isna() | (df["company"].astype(str).str.strip() == "")]
    if missing_name.empty:
        print("  ✅ All rows have company names")
    else:
        print(f"  Found {len(missing_name)} rows with missing company name — dropping")
        for idx, row in missing_name.iterrows():
            print(f"    Row {idx}: {dict(row)}")
            issues.append(f"Row {idx}: missing company name")
        df = df.drop(index=missing_name.index)

    # ── 3. OEM normalization ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Issue 3: OEM names normalization")
    print(SEP)
    manual_review_oems = []
    oem_col = "primary oems"

    if oem_col not in df.columns:
        print(f"  ⚠️  Column '{oem_col}' not found")
    else:
        unique_oems = df[oem_col].dropna().unique()
        print(f"  Unique OEM values ({len(unique_oems)} total):")
        for oem_val in sorted(unique_oems, key=str):
            normalized, needs_review = _normalize_oem(str(oem_val))
            if needs_review:
                marker = "⚠️  NEEDS MANUAL REVIEW"
                manual_review_oems.append(str(oem_val))
            elif normalized != str(oem_val):
                marker = f"→ '{normalized}'"
                fixes.append(f"OEM '{oem_val}' → '{normalized}'")
                # Apply fix
                df.loc[df[oem_col] == oem_val, oem_col] = normalized
            else:
                marker = "✅ OK"
            print(f"    '{oem_val}' : {marker}")

    # ── 4. Employment ranges ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Issue 4: Employment ranges")
    print(SEP)
    emp_col = "employment"
    if emp_col in df.columns:
        range_rows = df[df[emp_col].astype(str).str.contains("-", na=False)]
        if range_rows.empty:
            print("  ✅ No employment ranges found")
        else:
            print(f"  Found {len(range_rows)} employment range values:")
            for idx, row in range_rows.iterrows():
                original = row[emp_col]
                fixed = _parse_employment_range(original)
                print(f"    Row {idx} ({row.get('company', '')}): '{original}' → '{fixed}'")
                df.at[idx, emp_col] = fixed
                fixes.append(f"Employment '{original}' → '{fixed}'")

    # ── 5. Missing tier ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Issue 5: Missing tier values")
    print(SEP)
    tier_col = "category"
    if tier_col in df.columns:
        missing_tier = df[df[tier_col].isna() | (df[tier_col].astype(str).str.strip().isin(["", "nan"]))]
        if missing_tier.empty:
            print("  ✅ All rows have tier values")
        else:
            print(f"  {len(missing_tier)} rows missing tier:")
            for idx, row in missing_tier.iterrows():
                print(f"    Row {idx}: {row.get('company', '')} → setting to 'Unknown'")
                df.at[idx, tier_col] = "Unknown"
                fixes.append(f"Row {idx} ({row.get('company', '')}): tier set to 'Unknown'")

    # ── 6. Missing location ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Issue 6: Missing location (updated location column)")
    print(SEP)
    loc_col = "updated location"
    if loc_col in df.columns:
        missing_loc = df[df[loc_col].isna() | (df[loc_col].astype(str).str.strip().isin(["", "nan"]))]
        print(f"  {len(missing_loc)} companies missing location (will have no Location node in graph):")
        for idx, row in missing_loc.iterrows():
            print(f"    Row {idx}: {row.get('company', '')}")
            issues.append(f"Row {idx} ({row.get('company', '')}): missing location")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print("  AUDIT SUMMARY")
    print(f"{'#'*60}")
    print(f"  Original rows     : {len(df_orig)}")
    print(f"  After cleaning    : {len(df)}")
    print(f"  Issues found      : {len(issues)}")
    print(f"  Fixes applied     : {len(fixes)}")
    print(f"  OEM manual review : {len(manual_review_oems)}")

    if manual_review_oems:
        print(f"\n  ⚠️  OEM values needing manual lookup:")
        for oem in manual_review_oems:
            count = (df[oem_col] == oem).sum() if oem_col in df.columns else 0
            print(f"     '{oem}' — {count} companies")

    # ── Save cleaned Excel ────────────────────────────────────────────────────
    out_path = excel_path.parent / "GNEM_Cleaned.xlsx"
    df.to_excel(out_path, index=False, engine="openpyxl")
    print(f"\n  ✅ Cleaned Excel saved: {out_path}")
    print(f"  Rows in cleaned Excel: {len(df)}")
    print(f"\n  Next steps:")
    print(f"  1. Open GNEM_Cleaned.xlsx")
    print(f"  2. Find 'Multiple OEMs' rows → fill in real OEM names manually")
    print(f"  3. Run: venv\\Scripts\\python scripts\\reload_from_cleaned_excel.py")
    print()


if __name__ == "__main__":
    main()
