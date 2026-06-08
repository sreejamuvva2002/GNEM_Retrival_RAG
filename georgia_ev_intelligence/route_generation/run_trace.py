"""Run routing through context retrieval for one question and export XLSX."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.factory import (
    build_default_pipeline,
)
from georgia_ev_intelligence.runtime_pipeline.schemas import (
    ParentContext,
    RetrievedChildChunk,
)
from georgia_ev_intelligence.shared import config

from .route_service import RouteService, RouteTrace, build_default_route_service

EXCEL_CELL_LIMIT = 32767

NEXT_STEPS = {
    "no_retrieval": "Generate a direct response without KB retrieval.",
    "out_of_domain": "Reject or redirect the out-of-domain question.",
    "clarification_needed": "Ask the user for the missing clarification.",
    "exact_lookup": "Execute an exact entity lookup.",
    "structured_sql": "Execute a structured KB query.",
    "geo_search": "Execute a geographic KB search.",
    "keyword_search": "Retrieve keyword-matching document context.",
    "vector_search": "Retrieve semantically similar document context.",
    "hybrid_search": "Retrieve and merge keyword plus semantic context.",
    "disruption_analysis": "Retrieve evidence and run disruption analysis.",
}


@dataclass(frozen=True)
class TraceRun:
    route_trace: RouteTrace
    next_step: str
    retrieval_status: str
    retrieval_error: str
    parent_contexts: list[ParentContext]
    dense_children: list[RetrievedChildChunk]
    sparse_children: list[RetrievedChildChunk]
    retrieval_counts: dict[str, Any]
    pipeline_steps: list[dict[str, Any]]


def run_question_trace(
    question: str,
    route_service: RouteService | None = None,
    retrieval_pipeline_factory=build_default_pipeline,
) -> TraceRun:
    """Run one question through routing and the active context retriever."""
    service = route_service or build_default_route_service()
    route_trace = service.route_with_trace(question)
    final_route = route_trace.final_route
    next_step = NEXT_STEPS.get(final_route.route.value, "Hand off to the route executor.")
    steps = _routing_steps(route_trace)

    parent_contexts: list[ParentContext] = []
    dense_children: list[RetrievedChildChunk] = []
    sparse_children: list[RetrievedChildChunk] = []
    retrieval_counts: dict[str, Any] = {}
    retrieval_error = ""

    if not final_route.needs_kb_access:
        retrieval_status = "skipped"
        steps.append(_step(
            len(steps) + 1,
            "context_retrieval",
            retrieval_status,
            "Final route does not require KB access.",
        ))
    else:
        try:
            pipeline = retrieval_pipeline_factory()
            if hasattr(pipeline, "retrieve_with_sources"):
                result = pipeline.retrieve_with_sources(question)
                parent_contexts = result.parent_contexts
                dense_children = result.dense_children
                sparse_children = result.sparse_children
                retrieval_counts = _trace_counts(result.trace)
            else:
                parent_contexts = pipeline.retrieve(question)
                retrieval_counts = {"parent_context_count": len(parent_contexts)}
            retrieval_status = "completed"
            steps.append(_step(
                len(steps) + 1,
                "context_retrieval",
                retrieval_status,
                {
                    "executor": type(pipeline).__name__,
                    **retrieval_counts,
                },
            ))
        except Exception as exc:
            retrieval_status = "failed"
            retrieval_error = f"{type(exc).__name__}: {exc}"
            steps.append(_step(
                len(steps) + 1,
                "context_retrieval",
                retrieval_status,
                retrieval_error,
            ))

    return TraceRun(
        route_trace=route_trace,
        next_step=next_step,
        retrieval_status=retrieval_status,
        retrieval_error=retrieval_error,
        parent_contexts=parent_contexts,
        dense_children=dense_children,
        sparse_children=sparse_children,
        retrieval_counts=retrieval_counts,
        pipeline_steps=steps,
    )


def write_trace_workbook(trace_run: TraceRun, output_path: Path) -> None:
    """Write routing findings and retrieved contexts to separate XLSX sheets."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    route_trace = trace_run.route_trace
    final_route = route_trace.final_route

    summary = [{
        "question": final_route.question,
        "route": final_route.route.value,
        "route_source": final_route.route_source,
        "selected_router": route_trace.selected_router,
        "confidence": final_route.confidence,
        "validation_status": final_route.validation_status,
        "operation": final_route.operation,
        "next_step": trace_run.next_step,
        "needs_kb_access": final_route.needs_kb_access,
        "needs_document_retrieval": final_route.needs_document_retrieval,
        "retrieval_status": trace_run.retrieval_status,
        "retrieval_error": trace_run.retrieval_error,
        "parent_context_count": len(trace_run.parent_contexts),
        "dense_child_count": len(trace_run.dense_children),
        "sparse_child_count": len(trace_run.sparse_children),
        "llm_router_error": route_trace.llm_error,
    }]
    parent_rows = [
        {
            "rank": index,
            "record_id": parent.record_id,
            "source_row_id": parent.source_row_id,
            "context": parent.parent_chunk_text,
        }
        for index, parent in enumerate(trace_run.parent_contexts, start=1)
    ]

    sheets = {
        "summary": pd.DataFrame(summary),
        "pipeline_steps": pd.DataFrame(trace_run.pipeline_steps),
        "normalized_question": _key_value_frame(route_trace.normalized_question),
        "pre_route": _model_frame(route_trace.pre_route),
        "raw_route": _model_frame(route_trace.raw_route),
        "final_route": _model_frame(final_route),
        "retrieval_counts": _key_value_frame(trace_run.retrieval_counts),
        "parent_contexts": pd.DataFrame(
            parent_rows,
            columns=["rank", "record_id", "source_row_id", "context"],
        ),
        "dense_children": _children_frame(trace_run.dense_children),
        "sparse_children": _children_frame(trace_run.sparse_children),
    }

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, dataframe in sheets.items():
            _excel_safe_frame(dataframe).to_excel(writer, sheet_name=sheet_name, index=False)
        _format_workbook(writer.book)


def write_route_json(trace_run: TraceRun, output_path: Path) -> None:
    """Write the validated final route contract to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        trace_run.route_trace.final_route.model_dump_json(indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    output_path = args.output or _default_output_path()
    json_output_path = args.json_output or output_path.with_suffix(".json")
    trace_run = run_question_trace(args.question)
    write_trace_workbook(trace_run, output_path)
    write_route_json(trace_run, json_output_path)

    final_route = trace_run.route_trace.final_route
    print(f"Route: {final_route.route.value}")
    print(f"Next step: {trace_run.next_step}")
    print(f"Context retrieval: {trace_run.retrieval_status}")
    print(f"Saved trace workbook to {output_path}")
    print(f"Saved route JSON to {json_output_path}")
    return 0


def _routing_steps(route_trace: RouteTrace) -> list[dict[str, Any]]:
    pre_detail: Any = (
        route_trace.pre_route.model_dump(mode="json")
        if route_trace.pre_route is not None
        else "No deterministic pre-route matched; defer to LLM router."
    )
    route_detail = route_trace.raw_route.model_dump(mode="json")
    if route_trace.llm_error:
        route_detail = {"llm_error": route_trace.llm_error, "fallback_route": route_detail}
    return [
        _step(1, "normalize_question", "completed", route_trace.normalized_question),
        _step(2, "pre_router", "matched" if route_trace.pre_route else "deferred", pre_detail),
        _step(3, "route_selection", route_trace.selected_router, route_detail),
        _step(4, "route_validation", route_trace.final_route.validation_status,
              route_trace.final_route.model_dump(mode="json")),
        _step(5, "next_step", "identified",
              NEXT_STEPS.get(route_trace.final_route.route.value, "Hand off to route executor.")),
    ]


def _step(number: int, name: str, status: str, detail: Any) -> dict[str, Any]:
    return {
        "step_number": number,
        "stage": name,
        "status": status,
        "detail": _json_text(detail),
    }


def _trace_counts(trace: Any) -> dict[str, Any]:
    if trace is None:
        return {}
    return asdict(trace)


def _model_frame(model: Any) -> pd.DataFrame:
    if model is None:
        return _key_value_frame({})
    return _key_value_frame(model.model_dump(mode="json"))


def _key_value_frame(values: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"field": key, "value": _json_text(value)} for key, value in values.items()],
        columns=["field", "value"],
    )


def _children_frame(children: list[RetrievedChildChunk]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": index,
                "chunk_id": child.chunk_id,
                "parent_record_id": child.parent_record_id,
                "chunk_type": child.chunk_type,
                "metadata": _json_text(child.metadata),
            }
            for index, child in enumerate(children, start=1)
        ],
        columns=["rank", "chunk_id", "parent_record_id", "chunk_type", "metadata"],
    )


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)


def _excel_safe_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    safe = dataframe.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(_excel_safe_value)
    return safe


def _excel_safe_value(value: Any) -> Any:
    if not isinstance(value, str) or len(value) <= EXCEL_CELL_LIMIT:
        return value
    suffix = "\n[truncated to Excel cell limit]"
    return value[: EXCEL_CELL_LIMIT - len(suffix)] + suffix


def _format_workbook(workbook) -> None:
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
    return config.OUTPUTS_DIR / "route_traces" / f"{timestamp}_route_trace.xlsx"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace one question through routing and context retrieval into XLSX.",
    )
    parser.add_argument("--question", required=True, help="Question to route and retrieve.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output XLSX path.")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional route JSON path. Defaults beside the XLSX with the same filename stem.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
