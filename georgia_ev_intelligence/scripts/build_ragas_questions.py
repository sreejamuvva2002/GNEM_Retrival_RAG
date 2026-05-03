"""
scripts/build_ragas_questions.py
Reads outputs/questions_export.json and injects all 50 questions
into run_ragas_eval.py, replacing the SMOKE_QUESTIONS list.
Run once: venv\\Scripts\\python scripts\\build_ragas_questions.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src  = ROOT / "outputs" / "questions_export.json"
target = ROOT / "scripts" / "run_ragas_eval.py"

rows = json.loads(src.read_text(encoding="utf-8"))

lines = ["FIFTY_QUESTIONS: list[dict] = [\n"]
for r in rows:
    num      = str(r.get("Num","")).strip()
    cat      = str(r.get("Use Case Category","GENERAL")).strip()
    question = str(r.get("Question","")).strip()
    golden   = str(r.get("Human validated answers","")).strip()
    lines.append(
        f'    {{"id": "Q{num}", "category": {json.dumps(cat)}, '
        f'"question": {json.dumps(question)}, "golden": {json.dumps(golden)}}},\n'
    )
lines.append("]\n")

block = "".join(lines)

content = target.read_text(encoding="utf-8")

# Replace SMOKE_QUESTIONS list  (everything between the list header and the closing bracket)
import re
# Insert FIFTY_QUESTIONS after the SMOKE_QUESTIONS definition
if "FIFTY_QUESTIONS" in content:
    # Already injected — replace
    content = re.sub(
        r"FIFTY_QUESTIONS: list\[dict\] = \[.*?\]\n",
        block,
        content,
        flags=re.DOTALL,
    )
else:
    # Insert just before the main() function
    content = content.replace(
        "# ── Judge helpers",
        block + "\n# ── Judge helpers",
    )

# Also patch the main to use FIFTY_QUESTIONS instead of SMOKE_QUESTIONS[:n]
content = content.replace(
    "qs = SMOKE_QUESTIONS[:args.questions]",
    "qs = FIFTY_QUESTIONS[:args.questions]",
)

target.write_text(content, encoding="utf-8")
print(f"✅ Injected {len(rows)} questions into {target}")
print("Now run:")
print("  venv\\Scripts\\python scripts\\run_ragas_eval.py --questions 50")
