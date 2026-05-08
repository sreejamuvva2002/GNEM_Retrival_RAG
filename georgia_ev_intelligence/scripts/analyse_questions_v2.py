"""
FIXED: Cross-Analysis of 50 Questions vs GNEM Excel
Reads columns correctly:
  Col 0: Num
  Col 1: Use Case Category
  Col 2: Question        ← actual question
  Col 3: Human validated answers  ← ground truth answer
  Col 4: Answer from Web          ← web-sourced answer

Run: venv\\Scripts\\python scripts\\analyse_questions_v2.py
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from db_storage.kb_loader import _find_gnem_excel, _normalize_columns

SEP  = "=" * 72
HALF = "-" * 72

# ── Load files ────────────────────────────────────────────────────────────────
gnem_path = _find_gnem_excel()
gnem_df   = pd.read_excel(gnem_path, engine="openpyxl")
gnem_df   = _normalize_columns(gnem_df)

q_path = Path(gnem_path).parent / "Human validated 50 questions.xlsx"
q_df   = pd.read_excel(q_path, engine="openpyxl")

# Show raw columns first
print(f"\n{'#'*72}")
print("  50 QUESTIONS — COLUMN STRUCTURE")
print(f"{'#'*72}")
print(f"  Columns found: {list(q_df.columns)}")
print(f"  Total rows   : {len(q_df)}\n")
print(f"  First 3 rows sample:")
print(q_df.head(3).to_string())
print()

# Map correct column names
col_num      = q_df.columns[0]   # "Num"
col_category = q_df.columns[1]   # "Use Case Category"
col_question = q_df.columns[2]   # "Question"
col_answer   = q_df.columns[3]   # "Human validated answers"
col_web      = q_df.columns[4] if len(q_df.columns) > 4 else None   # "Answer from Web"

print(f"  Mapped → Question col: '{col_question}'")
print(f"  Mapped → Answer col  : '{col_answer}'")
print(f"  Mapped → Web col     : '{col_web}'\n")

# ── Print all 50 Q&A pairs ────────────────────────────────────────────────────
print(SEP)
print("  ALL 50 QUESTIONS + HUMAN VALIDATED ANSWERS")
print(SEP)
for _, row in q_df.iterrows():
    num      = row[col_num]
    category = str(row[col_category]).strip()
    question = str(row[col_question]).strip()
    answer   = str(row[col_answer]).strip()
    web_ans  = str(row[col_web]).strip() if col_web else "N/A"

    print(f"\n  [{num:02.0f}] [{category}]")
    print(f"  Q: {question}")
    print(f"  A (Human): {answer}")
    if web_ans not in ("nan", "N/A", ""):
        print(f"  A (Web)  : {web_ans}")
    print()

# ── Classify each question: GNEM-answerable vs Needs-web-data ────────────────
print(SEP)
print("  QUESTION SOURCE ANALYSIS: GNEM data vs External/Web data")
print(SEP)

# Keywords that indicate answer comes from GNEM columns only
GNEM_KEYWORDS = [
    "tier", "employment", "county", "ev supply chain role", "industry group",
    "product / service", "ev / battery relevant", "primary oems",
    "classification", "primary facility type", "multiple oems",
    "tier 1", "tier 2", "oem", "tier 2/3", "tier 1/2"
]

# Keywords in answers that suggest external/web knowledge (specific numbers, dates, $)
WEB_KEYWORDS = [
    "$", "billion", "million", "announced", "2023", "2024", "2025",
    "percent", "acres", "square feet", "opened", "unveiled", "broke ground"
]

gnem_answerable  = []
needs_web        = []
hybrid           = []

for _, row in q_df.iterrows():
    question = str(row[col_question]).lower()
    answer   = str(row[col_answer]).lower()
    web_ans  = str(row[col_web]).lower() if col_web else ""
    num      = row[col_num]
    cat      = str(row[col_category]).strip()

    has_gnem_signal = any(kw in answer for kw in GNEM_KEYWORDS)
    has_web_signal  = any(kw in answer or kw in web_ans for kw in WEB_KEYWORDS)

    if has_gnem_signal and not has_web_signal:
        gnem_answerable.append((num, cat, str(row[col_question])[:80]))
    elif has_web_signal and not has_gnem_signal:
        needs_web.append((num, cat, str(row[col_question])[:80]))
    else:
        hybrid.append((num, cat, str(row[col_question])[:80]))

print(f"\n  🟢 Questions answerable from GNEM alone ({len(gnem_answerable)}):")
for num, cat, q in gnem_answerable:
    print(f"     Q{num:02.0f}: {q}")

print(f"\n  🌐 Questions requiring web/external data ({len(needs_web)}):")
for num, cat, q in needs_web:
    print(f"     Q{num:02.0f}: {q}")

print(f"\n  🔀 Hybrid (GNEM + web both needed) ({len(hybrid)}):")
for num, cat, q in hybrid:
    print(f"     Q{num:02.0f}: {q}")

# ── GNEM Completeness ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  GNEM DATA COMPLETENESS")
print(SEP)
key_cols = {
    "company"              : "Company Name",
    "category"             : "Tier",
    "industry group"       : "Industry Group",
    "updated location"     : "Location",
    "primary oems"         : "Primary OEMs",
    "employment"           : "Employment",
    "product / service"    : "Products / Services",
    "ev / battery relevant": "EV Battery Relevant",
    "ev supply chain role" : "EV Supply Chain Role",
    "latitude"             : "Latitude",
    "longitude"            : "Longitude",
}
for col, label in key_cols.items():
    if col not in gnem_df.columns:
        print(f"  {label:<30} ⚠️  COLUMN NOT FOUND")
        continue
    total   = len(gnem_df)
    filled  = gnem_df[col].notna().sum()
    missing = total - filled
    multi   = 0
    if col == "primary oems":
        multi = (gnem_df[col].astype(str).str.strip().str.lower() == "multiple oems").sum()
        specific = filled - multi
        print(f"  {label:<30} {filled:>3}/{total} filled | {missing:>3} missing | {specific:>3} specific OEMs | {multi:>3} 'Multiple OEMs'")
    else:
        pct = (filled/total)*100
        print(f"  {label:<30} {filled:>3}/{total} filled | {missing:>3} missing  ({pct:.0f}%)")

# ── Save clean markdown artifact ──────────────────────────────────────────────
lines = [
    "# 50 Questions vs GNEM Data — Analysis Report\n",
    "## Key Finding: Where Did the Questions Come From?\n",
    f"- **{len(gnem_answerable)} questions** are answerable from GNEM Excel data alone\n",
    f"- **{len(needs_web)} questions** require external/web data (specific investments, job numbers, dates)\n",
    f"- **{len(hybrid)} questions** need both GNEM + web data\n",
    "\n## Data Completeness Summary\n",
    "| Field | Filled | Missing | Notes |\n",
    "|-------|--------|---------|-------|\n",
]
for col, label in key_cols.items():
    if col not in gnem_df.columns:
        lines.append(f"| {label} | N/A | N/A | Column missing |\n")
        continue
    filled  = int(gnem_df[col].notna().sum())
    missing = int(len(gnem_df) - filled)
    note    = ""
    if col == "primary oems":
        multi = int((gnem_df[col].astype(str).str.strip().str.lower() == "multiple oems").sum())
        note  = f"{multi} rows say 'Multiple OEMs' (vague)"
    lines.append(f"| {label} | {filled}/205 | {missing} | {note} |\n")

lines.append("\n## All 50 Questions\n")
lines.append("| # | Category | Question | Human Answer | Web Answer |\n")
lines.append("|---|----------|----------|--------------|------------|\n")
for _, row in q_df.iterrows():
    q   = str(row[col_question]).replace("|", "/").replace("\n", " ")
    a   = str(row[col_answer]).replace("|", "/").replace("\n", " ")
    w   = str(row[col_web] if col_web else "").replace("|", "/").replace("\n", " ")
    cat = str(row[col_category]).replace("|", "/")
    lines.append(f"| {row[col_num]} | {cat} | {q[:100]} | {a[:150]} | {w[:150]} |\n")

out = Path("scripts/gnem_analysis_report.md")
out.write_text("".join(lines), encoding="utf-8")
print(f"\n  ✅ Report saved: {out.resolve()}\n")
