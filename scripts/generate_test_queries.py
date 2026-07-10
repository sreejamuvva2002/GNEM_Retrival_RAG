#!/usr/bin/env python3
"""Generate synthetic test questions with a known expected route.

Thin CLI driver around ``georgia_ev_intelligence.eval.query_generator``. Each
question is grounded in real KB entity values (Normalized_kb.xlsx) and tagged
with the route its template targets, so ``scripts/eval_routing.py`` can score
routing accuracy without manual labeling.

Outputs:

  * data/questions_generated.csv           - question_id, question, expected_route, template_id
  * data/questions_generated_manifest.json - entity values substituted per question_id

Run from the project root:

    python scripts/generate_test_queries.py
    python scripts/generate_test_queries.py --count-per-route 25 --seed 7
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from georgia_ev_intelligence.eval.query_generator import DEFAULT_KB_PATH, generate_questions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic test questions with a known expected route.",
    )
    parser.add_argument(
        "--count-per-route",
        type=int,
        default=15,
        help="Target number of questions per route (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: %(default)s).")
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=DEFAULT_KB_PATH,
        help="Normalized_kb.xlsx path to draw entities from (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "questions_generated.csv",
        help="Output CSV path (default: %(default)s).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "questions_generated_manifest.json",
        help="Output manifest JSON path (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    questions = generate_questions(
        count_per_route=args.count_per_route, seed=args.seed, kb_path=args.kb_path
    )
    if not questions:
        print("ERROR: no questions were generated (empty KB?).")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["question_id", "question", "expected_route", "template_id"])
        manifest: dict[str, dict] = {}
        for idx, q in enumerate(questions, start=1):
            question_id = f"g{idx:03d}"
            writer.writerow([question_id, q.question, q.expected_route, q.template_id])
            manifest[question_id] = q.entities_used

    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    by_route = Counter(q.expected_route for q in questions)
    print(f"Wrote {len(questions)} questions to {args.output}")
    print(f"Wrote manifest to {args.manifest}")
    print("-" * 48)
    for route, count in sorted(by_route.items()):
        print(f"  {route}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
