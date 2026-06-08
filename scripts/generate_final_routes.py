#!/usr/bin/env python3
"""Generate validated final-route JSON for each of the 50 test questions.

Reads ``data/questions_50.csv`` (columns: ``question_id, question``), runs every
question through the EXISTING route-generation pipeline -
``build_default_route_service().route_with_trace(question).final_route`` - and
stores only the validated ``FinalRoute`` contract for each question.

This is a thin *batch driver* around the existing pipeline. It deliberately does
NOT (per project boundaries - see ``route_generation/__init__.py``):

  * call the route executor or any retrieval pipeline,
  * generate final answers,
  * run SQL / vector / keyword / geo search or disruption analysis,
  * store raw (untrusted) LLM route output as the main result,
  * redesign or re-implement the router or validator.

Outputs (UTF-8 JSONL, one record per question, overwritten each run):

  * outputs/final_routes.jsonl         - one record per successfully routed question
  * outputs/final_routes_failed.jsonl  - one record per question that raised an error

Run from the project root:

    python scripts/generate_final_routes.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# --- Make the package importable when run as `python scripts/...` from root. ---
# There is no editable install, so the project root (parent of scripts/) must be
# on sys.path before importing georgia_ev_intelligence.*.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Existing final-route pipeline, re-exported from the package. No redesign.
from georgia_ev_intelligence.route_generation import build_default_route_service

SCHEMA_VERSION = "v1"

QUESTION_ID_COLUMN = "question_id"
QUESTION_COLUMN = "question"

# The exact set/order of fields stored inside ``final_route`` (spec contract).
# The validated FinalRoute model also carries ``question`` and ``confidence``;
# those are intentionally dropped so each stored ``final_route`` matches the
# documented record format exactly. The router/validator are NOT changed - this
# is purely an output projection of the existing FinalRoute.
FINAL_ROUTE_FIELDS = (
    "route",
    "operation",
    "entities",
    "raw_filters",
    "resolved_filters",
    "requested_columns",
    "group_by",
    "sort_by",
    "limit",
    "query_focus",
    "needs_kb_access",
    "needs_document_retrieval",
    "missing_fields",
    "validation_status",
    "route_source",
    "clarification",
    "reason",
)


def _project_final_route(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only the documented ``final_route`` fields, in spec order."""
    return {key: payload[key] for key in FINAL_ROUTE_FIELDS if key in payload}


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing ``Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm(name: Any) -> str:
    return str(name).strip().lower()


def _resolve_column(columns: list[str], wanted: str) -> str:
    """Return the original column matching ``wanted`` (exact first, then case-insensitive)."""
    if wanted in columns:
        return wanted
    lookup = {_norm(c): c for c in columns}
    if wanted in lookup:
        return lookup[wanted]
    raise ValueError(
        f"Required column '{wanted}' not found. Available columns: {columns}"
    )


def _cell_or_none(value: Any) -> Any:
    """Return a trimmed string, or ``None`` for missing/blank cells."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def generate(
    input_path: Path,
    success_path: Path,
    failed_path: Path,
    limit: int | None = None,
) -> tuple[int, int, int]:
    """Route every question in ``input_path`` and write the JSONL outputs.

    Returns ``(n_ok, n_failed, n_skipped)``.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # Read as text; the CSV is only read, never written.
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    columns = [str(c) for c in df.columns]

    id_col = _resolve_column(columns, QUESTION_ID_COLUMN)
    q_col = _resolve_column(columns, QUESTION_COLUMN)
    print(f"Using columns: question_id={id_col!r}, question={q_col!r}")

    # Build the existing pipeline ONCE and reuse it for every question.
    service = build_default_route_service()

    success_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_failed = n_skipped = 0

    with success_path.open("w", encoding="utf-8") as ok_fh, \
            failed_path.open("w", encoding="utf-8") as fail_fh:
        for position, (_, row) in enumerate(df.iterrows(), start=1):
            if limit is not None and position > limit:
                break

            question = _cell_or_none(row.get(q_col))
            if question is None:
                n_skipped += 1
                continue

            # Trust the CSV's question_id; fall back to a positional id if blank.
            question_id = _cell_or_none(row.get(id_col)) or f"q{position:03d}"

            generated_at = _utc_now_iso()
            try:
                # The validator's trusted handoff contract - NOT the raw LLM route.
                final_route = service.route_with_trace(question).final_route

                # FinalRoute is a Pydantic model -> JSON-safe dict (enums -> str).
                # If a future pipeline returns a plain dict, serialize it directly.
                if hasattr(final_route, "model_dump"):
                    final_route_payload = final_route.model_dump(mode="json")
                else:
                    final_route_payload = dict(final_route)

                # Store only the documented final_route fields (spec contract).
                final_route_payload = _project_final_route(final_route_payload)

                record = {
                    "question_id": question_id,
                    "question": question,
                    "schema_version": SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "final_route": final_route_payload,
                }
                ok_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_ok += 1
                print(
                    f"[ok]     {question_id} -> {final_route_payload.get('route')} "
                    f"[{final_route_payload.get('validation_status')}]"
                )
            except Exception as exc:  # per-row isolation: never abort the batch
                record = {
                    "question_id": question_id,
                    "question": question,
                    "error": f"{type(exc).__name__}: {exc}",
                    "generated_at": generated_at,
                }
                fail_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_failed += 1
                print(f"[FAILED] {question_id}: {record['error']}")

    return n_ok, n_failed, n_skipped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate validated final-route JSON for the 50 test questions "
            "(router + validator only; no execution / no answers)."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "questions_50.csv",
        help="CSV of questions with columns question_id,question (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for the JSONL outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: process only the first N rows (smoke testing).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    success_path = args.output_dir / "final_routes.jsonl"
    failed_path = args.output_dir / "final_routes_failed.jsonl"

    print(f"Input:   {args.input}")
    print(f"Success: {success_path}")
    print(f"Failed:  {failed_path}")
    print("-" * 48)

    try:
        n_ok, n_failed, n_skipped = generate(
            input_path=args.input,
            success_path=success_path,
            failed_path=failed_path,
            limit=args.limit,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 2

    print("-" * 48)
    print(f"Routed OK : {n_ok}")
    print(f"Failed    : {n_failed}")
    print(f"Skipped   : {n_skipped} (empty question rows)")
    print(f"Wrote {success_path}")
    print(f"Wrote {failed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
