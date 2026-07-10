#!/usr/bin/env python3
"""LLM-as-judge scoring of generated answers for faithfulness and relevance.

Reads an execution-results JSONL (as produced by
``scripts/run_execution_from_routes.py --use-llm``) and scores each
question/evidence/answer triple with
``georgia_ev_intelligence.eval.answer_judge.judge_answer`` - a separate LLM
call that never sees or modifies the original answer, only grades it. Does
NOT call the router or executor.

Outputs (UTF-8, overwritten each run):

  * outputs/answer_eval_results.jsonl  - one record per question (scores + notes)
  * outputs/answer_eval_summary.json   - mean scores overall + per-route, worst-N list

Run from the project root:

    python scripts/eval_answers.py
    python scripts/eval_answers.py --input outputs/generated_eval/execution_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from georgia_ev_intelligence.eval.answer_judge import judge_answer
from georgia_ev_intelligence.route_execution.answer_formatter import _grounded_evidence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate(
    input_path: Path,
    results_path: Path,
    summary_path: Path,
    limit: int | None = None,
    worst_n: int = 5,
) -> dict[str, Any]:
    records = _read_records(input_path)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    scored: list[dict[str, Any]] = []
    per_route_faith: dict[str, list[int]] = defaultdict(list)
    per_route_relev: dict[str, list[int]] = defaultdict(list)
    n_judged = n_skipped = n_errors = 0

    with results_path.open("w", encoding="utf-8") as fh:
        for position, record in enumerate(records, start=1):
            if limit is not None and position > limit:
                break

            question_id = record.get("question_id") or f"a{position:03d}"
            question = record.get("question", "")
            answer = record.get("answer", "")
            evidence = record.get("evidence", {}) or {}
            final_route = record.get("final_route", {}) or {}
            route = final_route.get("route", "unknown")

            if not answer:
                n_skipped += 1
                continue

            # Judge against the SAME whitelisted evidence the answer generator
            # was grounded on (answer_formatter._grounded_evidence), not the
            # raw evidence dict — the raw dict also carries sql/sql_params/
            # sql_display and lat/long noise the answer model never saw, which
            # would unfairly penalize faithfulness against evidence it wasn't
            # actually given.
            grounded_evidence = _grounded_evidence(evidence)

            out: dict[str, Any]
            try:
                judgment = judge_answer(question, grounded_evidence, answer)
                out = {
                    "question_id": question_id,
                    "question": question,
                    "route": route,
                    "answer": answer,
                    "faithfulness": judgment.faithfulness,
                    "relevance": judgment.relevance,
                    "notes": judgment.notes,
                    "generated_at": _utc_now_iso(),
                }
                per_route_faith[route].append(judgment.faithfulness)
                per_route_relev[route].append(judgment.relevance)
                scored.append(out)
                n_judged += 1
                print(
                    f"[ok]     {question_id} route={route} "
                    f"faithfulness={judgment.faithfulness} relevance={judgment.relevance}"
                )
            except Exception as exc:  # per-row isolation: never abort the batch
                out = {
                    "question_id": question_id,
                    "question": question,
                    "route": route,
                    "answer": answer,
                    "error": f"{type(exc).__name__}: {exc}",
                    "generated_at": _utc_now_iso(),
                }
                n_errors += 1
                print(f"[FAILED] {question_id}: {out['error']}")

            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    worst = sorted(scored, key=lambda r: r["faithfulness"] + r["relevance"])[:worst_n]

    summary = {
        "n_judged": n_judged,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
        "mean_faithfulness": _mean([r["faithfulness"] for r in scored]),
        "mean_relevance": _mean([r["relevance"] for r in scored]),
        "per_route_mean_faithfulness": {r: _mean(v) for r, v in sorted(per_route_faith.items())},
        "per_route_mean_relevance": {r: _mean(v) for r, v in sorted(per_route_relev.items())},
        "worst_n": [
            {
                "question_id": r["question_id"],
                "question": r["question"],
                "route": r["route"],
                "faithfulness": r["faithfulness"],
                "relevance": r["relevance"],
                "notes": r["notes"],
            }
            for r in worst
        ],
        "generated_at": _utc_now_iso(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge scoring of generated answers for faithfulness and relevance.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "execution_results.jsonl",
        help="Execution-results JSONL to grade (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for the JSON/JSONL outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: judge only the first N records (smoke testing).",
    )
    parser.add_argument(
        "--worst-n",
        type=int,
        default=5,
        help="Number of lowest-scoring answers to list in the summary (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results_path = args.output_dir / "answer_eval_results.jsonl"
    summary_path = args.output_dir / "answer_eval_summary.json"

    print(f"Input:   {args.input}")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print("-" * 48)

    try:
        summary = evaluate(
            input_path=args.input,
            results_path=results_path,
            summary_path=summary_path,
            limit=args.limit,
            worst_n=args.worst_n,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    print("-" * 48)
    mf, mr = summary["mean_faithfulness"], summary["mean_relevance"]
    print(f"Judged           : {summary['n_judged']}")
    print(f"Skipped (no ans) : {summary['n_skipped']}")
    print(f"Errors           : {summary['n_errors']}")
    print(f"Mean faithfulness: {mf:.2f}" if mf is not None else "Mean faithfulness: n/a")
    print(f"Mean relevance   : {mr:.2f}" if mr is not None else "Mean relevance   : n/a")
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
