"""Execute final routes and export every retrieved result plus SQL to XLSX.

Run from the project root:

    python -m georgia_ev_intelligence.route_execution.export_retrieved_contexts
"""
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from .executor import execute_route
from .schemas import STATUS_FAILED, ExecutionResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "final_routes.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXCEL_CELL_LIMIT = 32767

_STATE_NAMES = {
    "alabama": "Alabama",
    "alaska": "Alaska",
    "arizona": "Arizona",
    "arkansas": "Arkansas",
    "california": "California",
    "colorado": "Colorado",
    "connecticut": "Connecticut",
    "delaware": "Delaware",
    "florida": "Florida",
    "georgia": "Georgia",
    "hawaii": "Hawaii",
    "idaho": "Idaho",
    "illinois": "Illinois",
    "indiana": "Indiana",
    "iowa": "Iowa",
    "kansas": "Kansas",
    "kentucky": "Kentucky",
    "louisiana": "Louisiana",
    "maine": "Maine",
    "maryland": "Maryland",
    "massachusetts": "Massachusetts",
    "michigan": "Michigan",
    "minnesota": "Minnesota",
    "mississippi": "Mississippi",
    "missouri": "Missouri",
    "montana": "Montana",
    "nebraska": "Nebraska",
    "nevada": "Nevada",
    "new hampshire": "New Hampshire",
    "new jersey": "New Jersey",
    "new mexico": "New Mexico",
    "new york": "New York",
    "north carolina": "North Carolina",
    "north dakota": "North Dakota",
    "ohio": "Ohio",
    "oklahoma": "Oklahoma",
    "oregon": "Oregon",
    "pennsylvania": "Pennsylvania",
    "rhode island": "Rhode Island",
    "south carolina": "South Carolina",
    "south dakota": "South Dakota",
    "tennessee": "Tennessee",
    "texas": "Texas",
    "utah": "Utah",
    "vermont": "Vermont",
    "virginia": "Virginia",
    "washington": "Washington",
    "west virginia": "West Virginia",
    "wisconsin": "Wisconsin",
    "wyoming": "Wyoming",
    "district of columbia": "District of Columbia",
    "unknown": "Unknown",
}

_STATE_ABBREVIATIONS = {
    "al": "Alabama",
    "ak": "Alaska",
    "az": "Arizona",
    "ar": "Arkansas",
    "ca": "California",
    "co": "Colorado",
    "ct": "Connecticut",
    "de": "Delaware",
    "fl": "Florida",
    "ga": "Georgia",
    "hi": "Hawaii",
    "id": "Idaho",
    "il": "Illinois",
    "in": "Indiana",
    "ia": "Iowa",
    "ks": "Kansas",
    "ky": "Kentucky",
    "la": "Louisiana",
    "me": "Maine",
    "md": "Maryland",
    "ma": "Massachusetts",
    "mi": "Michigan",
    "mn": "Minnesota",
    "ms": "Mississippi",
    "mo": "Missouri",
    "mt": "Montana",
    "ne": "Nebraska",
    "nv": "Nevada",
    "nh": "New Hampshire",
    "nj": "New Jersey",
    "nm": "New Mexico",
    "ny": "New York",
    "nc": "North Carolina",
    "nd": "North Dakota",
    "oh": "Ohio",
    "ok": "Oklahoma",
    "or": "Oregon",
    "pa": "Pennsylvania",
    "ri": "Rhode Island",
    "sc": "South Carolina",
    "sd": "South Dakota",
    "tn": "Tennessee",
    "tx": "Texas",
    "ut": "Utah",
    "vt": "Vermont",
    "va": "Virginia",
    "wa": "Washington",
    "wv": "West Virginia",
    "wi": "Wisconsin",
    "wy": "Wyoming",
    "dc": "District of Columbia",
}


@dataclass
class ExecutedRoute:
    question_id: str
    question: str
    original_route: dict[str, Any]
    executed_route: dict[str, Any]
    result: ExecutionResult
    generated_at: str


def read_routes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Routes file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return records


def normalize_state_filter(final_route: dict[str, Any]) -> dict[str, Any]:
    """Upgrade legacy state-as-location routes to the real ``state`` column."""
    route = deepcopy(final_route)
    filters = dict(route.get("resolved_filters") or {})
    state = _state_from_raw_filters(route.get("raw_filters") or [])

    location_filter = filters.get("updated_location")
    if state is None and isinstance(location_filter, dict):
        state = _single_state_from_filter(location_filter)

    if state is None:
        return route

    filters["state"] = {"operator": "EQUALS", "value": state}
    if isinstance(location_filter, dict) and _filter_contains_only_states(location_filter):
        filters.pop("updated_location", None)
    route["resolved_filters"] = filters

    if final_route.get("resolved_filters") != filters:
        actions = list(route.get("validation_actions") or [])
        actions.append("normalized legacy state location filter to state column for execution")
        route["validation_actions"] = actions
    return route


def execute_routes(
    records: list[dict[str, Any]],
    *,
    use_llm: bool = False,
    limit: int | None = None,
    executor: Callable[..., ExecutionResult] = execute_route,
) -> list[ExecutedRoute]:
    executions: list[ExecutedRoute] = []

    for position, record in enumerate(records, start=1):
        if limit is not None and position > limit:
            break

        original_route = record.get("final_route") or record
        if not isinstance(original_route, dict):
            original_route = {}
        executed_route = normalize_state_filter(original_route)
        question_id = str(record.get("question_id") or f"q{position:03d}")
        question = str(
            record.get("question")
            or executed_route.get("question")
            or ""
        )
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            result = executor(executed_route, use_llm=use_llm)
        except Exception as exc:
            result = ExecutionResult.failure(
                route=str(executed_route.get("route")),
                reason=f"{type(exc).__name__}: {exc}",
            )

        executions.append(
            ExecutedRoute(
                question_id=question_id,
                question=question,
                original_route=original_route,
                executed_route=executed_route,
                result=result,
                generated_at=generated_at,
            )
        )
        marker = "FAILED" if result.status == STATUS_FAILED else "ok"
        print(f"[{marker:<6}] {question_id} -> {executed_route.get('route')}")

    return executions


def write_workbook(executions: list[ExecutedRoute], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheets = {
        "summary": pd.DataFrame(_summary_rows(executions)),
        "sql_commands": pd.DataFrame(_all_sql_rows(executions)),
        "structured_data": pd.DataFrame(_all_structured_rows(executions)),
        "retrieved_contexts": pd.DataFrame(_all_context_rows(executions)),
        "final_routes": pd.DataFrame(_route_rows(executions)),
    }

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, dataframe in sheets.items():
            _excel_safe_frame(dataframe).to_excel(writer, sheet_name=sheet_name, index=False)
        _format_workbook(writer.book)


def run(
    input_path: Path,
    output_path: Path,
    *,
    use_llm: bool = False,
    limit: int | None = None,
    executor: Callable[..., ExecutionResult] = execute_route,
) -> tuple[int, int]:
    records = read_routes(input_path)
    executions = execute_routes(records, use_llm=use_llm, limit=limit, executor=executor)
    write_workbook(executions, output_path)
    successes = sum(item.result.status != STATUS_FAILED for item in executions)
    return successes, len(executions) - successes


def _summary_rows(executions: list[ExecutedRoute]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in executions:
        evidence = item.result.evidence or {}
        sql_rows = _sql_rows(item)
        rows.append({
            "question_id": item.question_id,
            "question": item.question,
            "route": item.executed_route.get("route"),
            "status": item.result.status,
            "evidence_type": evidence.get("type"),
            "retrieved_count": _retrieved_count(evidence),
            "sql_command_count": len(sql_rows),
            "answer": item.result.answer,
            "error": item.result.error or "",
            "generated_at": item.generated_at,
        })
    return rows


def _all_sql_rows(executions: list[ExecutedRoute]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in executions:
        rows.extend(_sql_rows(item))
    return rows


def _sql_rows(item: ExecutedRoute) -> list[dict[str, Any]]:
    evidence = item.result.evidence or {}
    commands: list[dict[str, Any]] = []

    if evidence.get("sql"):
        commands.append({
            "label": "primary",
            "sql": evidence.get("sql"),
            "sql_params": evidence.get("sql_params"),
            "sql_display": evidence.get("sql_display") or evidence.get("sql"),
        })
    if evidence.get("structured_sql"):
        commands.append({
            "label": "structured_before_fallback",
            "sql": evidence.get("structured_sql"),
            "sql_params": evidence.get("structured_sql_params"),
            "sql_display": (
                evidence.get("structured_sql_display")
                or evidence.get("structured_sql")
            ),
        })
    commands.extend(
        command
        for command in (evidence.get("sql_commands") or [])
        if isinstance(command, dict)
    )

    return [
        {
            "question_id": item.question_id,
            "question": item.question,
            "route": item.executed_route.get("route"),
            "command_number": index,
            "label": command.get("label") or "query",
            "sql_template": command.get("sql") or "",
            "sql_params": _json_text(command.get("sql_params")),
            "sql_display": command.get("sql_display") or command.get("sql") or "",
        }
        for index, command in enumerate(commands, start=1)
    ]


def _all_structured_rows(executions: list[ExecutedRoute]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in executions:
        evidence = item.result.evidence or {}
        evidence_type = evidence.get("type")
        retrieved: list[dict[str, Any]] = []

        if evidence_type == "structured_rows":
            retrieved = evidence.get("rows") or []
        elif evidence_type in {"group_counts", "group_aggregates"}:
            retrieved = evidence.get("groups") or []
        elif evidence_type == "count":
            retrieved = [{"count": evidence.get("count")}]
        elif evidence_type == "geo_results":
            retrieved = evidence.get("rows") or []
        elif evidence_type == "ranked_alternatives":
            retrieved = evidence.get("alternatives") or []

        for index, record in enumerate(retrieved, start=1):
            row = {
                "question_id": item.question_id,
                "question": item.question,
                "route": item.executed_route.get("route"),
                "evidence_type": evidence_type,
                "row_number": index,
            }
            if isinstance(record, dict):
                row.update({key: _cell_value(value) for key, value in record.items()})
            else:
                row["value"] = _cell_value(record)
            rows.append(row)
    return rows


def _all_context_rows(executions: list[ExecutedRoute]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in executions:
        evidence = item.result.evidence or {}
        if evidence.get("type") != "document_chunks":
            continue

        chunks_by_parent: dict[str, list[dict[str, Any]]] = {}
        for chunk in evidence.get("chunks") or []:
            if not isinstance(chunk, dict):
                continue
            parent_id = str(chunk.get("parent_record_id") or "")
            chunks_by_parent.setdefault(parent_id, []).append(chunk)

        for index, parent in enumerate(evidence.get("parents") or [], start=1):
            if not isinstance(parent, dict):
                parent = {"text": parent}
            parent_id = str(parent.get("parent_record_id") or "")
            matching_chunks = chunks_by_parent.get(parent_id, [])
            rows.append({
                "question_id": item.question_id,
                "question": item.question,
                "route": item.executed_route.get("route"),
                "context_rank": index,
                "parent_record_id": parent_id,
                "source_row_id": parent.get("source_row_id"),
                "chunk_ids": _json_text([chunk.get("chunk_id") for chunk in matching_chunks]),
                "chunk_types": _json_text([chunk.get("chunk_type") for chunk in matching_chunks]),
                "context": parent.get("text") or "",
            })
    return rows


def _route_rows(executions: list[ExecutedRoute]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": item.question_id,
            "question": item.question,
            "route": item.executed_route.get("route"),
            "original_final_route_json": _json_text(item.original_route),
            "executed_final_route_json": _json_text(item.executed_route),
        }
        for item in executions
    ]


def _retrieved_count(evidence: dict[str, Any]) -> int:
    evidence_type = evidence.get("type")
    if evidence_type == "count":
        return int(evidence.get("count") or 0)
    if evidence_type in {"group_counts", "group_aggregates"}:
        return len(evidence.get("groups") or [])
    if evidence_type in {"structured_rows", "geo_results"}:
        return len(evidence.get("rows") or [])
    if evidence_type == "document_chunks":
        return len(evidence.get("parents") or [])
    if evidence_type == "ranked_alternatives":
        return len(evidence.get("alternatives") or [])
    return 0


def _state_from_raw_filters(raw_filters: list[Any]) -> str | None:
    for raw_filter in raw_filters:
        if not isinstance(raw_filter, dict):
            continue
        if str(raw_filter.get("field_hint") or "").lower() != "state":
            continue
        for key in ("raw_value", "source_text"):
            state = _canonical_state(raw_filter.get(key))
            if state is not None:
                return state
    return None


def _single_state_from_filter(spec: dict[str, Any]) -> str | None:
    value = spec.get("value")
    if isinstance(value, (list, tuple)):
        states = [_canonical_state(item) for item in value]
        states = [state for state in states if state is not None]
        return states[0] if len(states) == 1 else None
    return _canonical_state(value)


def _filter_contains_only_states(spec: dict[str, Any]) -> bool:
    value = spec.get("value")
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    return bool(values) and all(_canonical_state(item) is not None for item in values)


def _canonical_state(value: Any) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[^a-z ]", "", str(value).strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    return _STATE_NAMES.get(normalized) or _STATE_ABBREVIATIONS.get(normalized)


def _cell_value(value: Any) -> Any:
    if isinstance(value, set):
        return _json_text(sorted(value, key=str))
    if isinstance(value, tuple):
        return _json_text(list(value))
    if isinstance(value, (dict, list)):
        return _json_text(value)
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value, key=str)
    return str(value)


def _excel_safe_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    safe = dataframe.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(_excel_safe_value)
    return safe


def _excel_safe_value(value: Any) -> Any:
    value = _cell_value(value)
    if not isinstance(value, str) or len(value) <= EXCEL_CELL_LIMIT:
        return value
    suffix = "\n[truncated to Excel cell limit]"
    return value[: EXCEL_CELL_LIMIT - len(suffix)] + suffix


def _format_workbook(workbook: Any) -> None:
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
        for column_cells in worksheet.columns:
            letter = get_column_letter(column_cells[0].column)
            max_length = max(len(str(cell.value or "")) for cell in column_cells[:100])
            worksheet.column_dimensions[letter].width = min(max(max_length + 2, 12), 80)
            for cell in column_cells:
                cell.alignment = Alignment(vertical="top", wrap_text=True)


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"{timestamp}_retrieved_contexts.xlsx"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute final routes and export all retrieved data plus SQL commands to XLSX.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Final routes JSONL (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output XLSX path (default: timestamped file in outputs/).",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Also call the local answer LLM after retrieval.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Execute only the first N routes.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_path = args.output or _default_output_path()
    print(f"Input:  {args.input}")
    print(f"Output: {output_path}")

    try:
        successes, failures = run(
            args.input,
            output_path,
            use_llm=args.use_llm,
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"Executed OK: {successes}")
    print(f"Failed:      {failures}")
    print(f"Saved retrieved contexts and SQL commands to {output_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
