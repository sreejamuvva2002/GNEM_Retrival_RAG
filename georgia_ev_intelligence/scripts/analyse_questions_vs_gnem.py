"""
Cross-Analysis: 50 Questions vs GNEM Excel
Answers:
  1. Were the 50 questions created from GNEM data or from external/internet sources?
  2. What data is missing in GNEM for each company?
  3. Which questions can our pipeline answer vs cannot?

Run: venv\\Scripts\\python scripts\\analyse_questions_vs_gnem.py
Saves: analysis_report.md (open in any markdown viewer)
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from phase1_extraction.kb_loader import _find_gnem_excel, _normalize_columns

SEP = "=" * 70
HALF = "-" * 70

# ── Load GNEM Excel ───────────────────────────────────────────────────────────
gnem_path = _find_gnem_excel()
gnem_df   = pd.read_excel(gnem_path, engine="openpyxl")
gnem_df   = _normalize_columns(gnem_df)

# ── Load 50 Questions Excel ───────────────────────────────────────────────────
q_path = Path(gnem_path).parent / "Human validated 50 questions.xlsx"
q_df   = pd.read_excel(q_path, engine="openpyxl")
q_df.columns = [str(c).strip() for c in q_df.columns]

print(f"\n{'#'*70}")
print("  ANALYSIS: 50 QUESTIONS vs GNEM EXCEL")
print(f"{'#'*70}")
print(f"\n  GNEM Excel   : {gnem_path.name}  ({len(gnem_df)} rows)")
print(f"  Questions    : {q_path.name}  ({len(q_df)} rows)")
print(f"\n  Questions columns: {list(q_df.columns)}")
print(f"  GNEM columns: {list(gnem_df.columns)}\n")

# ── Print all 50 questions with their answers ─────────────────────────────────
print(SEP)
print("  ALL 50 QUESTIONS + ANSWERS (full content)")
print(SEP)
for i, row in q_df.iterrows():
    print(f"\n  Q{i+1:02d}: {row.iloc[0] if len(row) > 0 else 'N/A'}")
    if len(row) > 1:
        print(f"  A:   {row.iloc[1]}")
    if len(row) > 2:
        print(f"  Cat: {row.iloc[2]}")
    print()

# ── GNEM Data Completeness Analysis ──────────────────────────────────────────
print(SEP)
print("  GNEM DATA COMPLETENESS — Per Column")
print(SEP)

key_columns = [
    "company", "category", "industry group", "updated location",
    "primary oems", "employment", "product / service",
    "ev / battery relevant", "ev supply chain role",
    "latitude", "longitude", "classification method",
]

gnem_companies = list(gnem_df["company"].dropna().astype(str).str.strip())

completeness = {}
for col in key_columns:
    if col not in gnem_df.columns:
        completeness[col] = {"filled": 0, "missing": len(gnem_df), "pct": 0}
        continue
    filled  = gnem_df[col].notna().sum()
    missing = gnem_df[col].isna().sum()
    # Also count "Multiple OEMs" as incomplete for primary_oems
    if col == "primary oems":
        multi = (gnem_df[col].astype(str).str.lower() == "multiple oems").sum()
        print(f"  {col:<35} {filled:>3} filled | {missing:>3} missing | {multi:>3} 'Multiple OEMs'")
    else:
        print(f"  {col:<35} {filled:>3} filled | {missing:>3} missing")
    completeness[col] = {"filled": int(filled), "missing": int(missing)}

# ── Per-Company Missing Data ──────────────────────────────────────────────────
print(f"\n{SEP}")
print("  PER-COMPANY DATA GAPS")
print(SEP)

company_gaps = []
for _, row in gnem_df.iterrows():
    name = str(row.get("company", "")).strip()
    if not name or name.lower() == "nan":
        continue
    
    missing_fields = []
    # Check each critical field
    checks = {
        "employment":        row.get("employment"),
        "location":          row.get("updated location"),
        "primary_oems":      row.get("primary oems"),
        "latitude":          row.get("latitude"),
        "longitude":         row.get("longitude"),
        "ev_role":           row.get("ev supply chain role"),
        "products_services": row.get("product / service"),
        "ev_battery_rel":    row.get("ev / battery relevant"),
    }
    for field, val in checks.items():
        is_empty = (val is None) or (isinstance(val, float) and pd.isna(val)) or (str(val).strip().lower() in ("", "nan", "none"))
        if is_empty:
            missing_fields.append(field)
        elif field == "primary_oems" and str(val).strip().lower() == "multiple oems":
            missing_fields.append("primary_oems(vague)")

    tier = str(row.get("category", "")).strip()
    county_raw = str(row.get("updated location", "")).strip()
    county = county_raw.split(",")[-1].strip() if "," in county_raw else county_raw

    company_gaps.append({
        "name": name,
        "tier": tier,
        "county": county,
        "missing_count": len(missing_fields),
        "missing_fields": missing_fields,
    })

# Sort by most missing fields first
company_gaps.sort(key=lambda x: x["missing_count"], reverse=True)

print(f"\n  Companies with MOST missing data (top 20):")
print(f"  {'Company':<40} {'Tier':<15} {'Missing Fields'}")
print(f"  {'-'*40} {'-'*15} {'-'*30}")
for gap in company_gaps[:20]:
    fields_str = ", ".join(gap["missing_fields"])
    print(f"  {gap['name'][:39]:<40} {gap['tier'][:14]:<15} {fields_str}")

# Companies with no missing data
perfect = [g for g in company_gaps if g["missing_count"] == 0]
print(f"\n  ✅ {len(perfect)} companies with COMPLETE data (no missing fields)")

# ── Save full report to markdown ──────────────────────────────────────────────
report_lines = []
report_lines.append("# GNEM Data Analysis Report\n")
report_lines.append("## 1. Column Completeness\n")
report_lines.append("| Column | Filled | Missing |")
report_lines.append("|--------|--------|---------|")
for col in key_columns:
    c = completeness.get(col, {})
    report_lines.append(f"| {col} | {c.get('filled',0)} | {c.get('missing',0)} |")

report_lines.append("\n## 2. Per-Company Data Gaps\n")
report_lines.append("| Company | Tier | County | Missing Count | Missing Fields |")
report_lines.append("|---------|------|--------|---------------|----------------|")
for gap in company_gaps:
    fields_str = ", ".join(gap["missing_fields"]) if gap["missing_fields"] else "None"
    report_lines.append(
        f"| {gap['name']} | {gap['tier']} | {gap['county']} | "
        f"{gap['missing_count']} | {fields_str} |"
    )

report_lines.append("\n## 3. All 50 Questions\n")
report_lines.append("| # | Question | Answer | Category |")
report_lines.append("|---|----------|--------|----------|")
for i, row in q_df.iterrows():
    q   = str(row.iloc[0] if len(row) > 0 else "").replace("|", "/")
    a   = str(row.iloc[1] if len(row) > 1 else "").replace("|", "/")
    cat = str(row.iloc[2] if len(row) > 2 else "").replace("|", "/")
    report_lines.append(f"| {i+1} | {q[:80]} | {a[:120]} | {cat} |")

# Save
report_path = Path("scripts/gnem_analysis_report.md")
report_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"\n  ✅ Full report saved: {report_path}")
print(f"  Open it in VS Code to read the formatted markdown table\n")
