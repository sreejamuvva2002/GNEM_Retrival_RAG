"""Extract the company list from the GNEM Excel file into companies.csv.

Usage:
    python scripts/extract_companies.py [--xlsx PATH] [--out PATH]

Defaults to data/GNEM_Excel_Data.xlsx (the uploaded company sheet). Handles both
the "Location" and "Updated Location" column names seen across GNEM exports.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
DEFAULT_XLSX = PKG_ROOT / "data" / "GNEM_Excel_Data.xlsx"
DEFAULT_OUT = PKG_ROOT / "data" / "companies.csv"

# Output column -> acceptable source column names (first match wins)
COLUMN_MAP = {
    "company": ["Company"],
    "category": ["Category"],
    "location": ["Location", "Updated Location"],
    "address": ["Address"],
    "industry_group": ["Industry Group"],
    "ev_role": ["EV Supply Chain Role"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    wb = openpyxl.load_workbook(args.xlsx, read_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(rows)]

    idx: dict[str, int] = {}
    for out_col, candidates in COLUMN_MAP.items():
        for cand in candidates:
            if cand in header:
                idx[out_col] = header.index(cand)
                break
    if "company" not in idx:
        raise SystemExit(f"'Company' column not found in {args.xlsx}; header was: {header}")

    seen: set[str] = set()
    out_rows: list[dict[str, str]] = []
    for row in rows:
        company = row[idx["company"]]
        if not company or not str(company).strip():
            continue
        name = str(company).strip()
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        record = {}
        for out_col in COLUMN_MAP:
            if out_col == "company":
                record[out_col] = name
                continue
            val = row[idx[out_col]] if out_col in idx else None
            record[out_col] = str(val).strip() if val is not None else ""
        out_rows.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMN_MAP))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} unique companies to {args.out}")


if __name__ == "__main__":
    main()
