"""Convert a baseline JSONL run folder into a RAGAS-compatible Excel workbook.

The output workbook has two sheets that the existing evaluate_ragas_ollama.py reads:
  - responses : Question + one column per (model, pipeline) combination
  - retrieval : question, rank, chunk_type, text  (one row per context chunk)

Usage:
    python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.build_ragas_report \
        --run-dir georgia_ev_intelligence/outputs/baselines/20260520_143000

    # Filter to specific pipelines or models:
    python -m ... --pipelines rag_only hybrid_rag --models qwen2.5:7b llama3.1:8b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _column_name(model: str, pipeline: str) -> str:
    safe_model = model.replace(":", "_").replace("/", "_")
    return f"{safe_model}__{pipeline}"


def build_responses_sheet(
    records_by_col: dict[str, list[dict]],
    questions: list[str],
) -> pd.DataFrame:
    """One row per question, one column per (model, pipeline) combination."""
    df = pd.DataFrame({"Question": questions})
    for col_name, records in records_by_col.items():
        answer_map = {r["question"]: r["answer"] for r in records}
        df[col_name] = df["Question"].map(answer_map).fillna("")
    return df


def build_retrieval_sheet(
    records_by_col: dict[str, list[dict]],
    questions: list[str],
) -> pd.DataFrame:
    """One row per (question, rank, chunk) for all retrieval-based pipelines.

    pretrained_only has no contexts and is excluded.
    For direct_kb, all 205 KB rows are included with chunk_type='kb_record'.
    """
    rows: list[dict] = []
    seen: set[tuple] = set()

    for col_name, records in records_by_col.items():
        pipeline = records[0]["pipeline"] if records else ""
        if pipeline == "pretrained_only":
            continue

        chunk_type = "kb_record" if pipeline == "direct_kb" else "parent_chunk"

        for record in records:
            question = record["question"]
            for rank, context_text in enumerate(record.get("contexts", []), start=1):
                key = (question, pipeline, rank)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "question": question,
                    "pipeline": pipeline,
                    "rank": rank,
                    "chunk_type": chunk_type,
                    "text": context_text,
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["question", "pipeline", "rank", "chunk_type", "text"]
    )


def build_golden_answers_sheet(questions: list[str], ground_truths: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"question": questions, "ground_truth": ground_truths})


def build_run_info_sheet(run_dir: Path, config_data: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "run_dir": str(run_dir.resolve()),
        **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in config_data.items()},
    }])


def write_workbook(output_path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for col in worksheet.columns:
                max_len = max(
                    (len(str(cell.value or "")) for cell in col),
                    default=10,
                )
                col_letter = col[0].column_letter
                worksheet.column_dimensions[col_letter].width = min(max_len + 2, 80)
    print(f"Saved RAGAS report: {output_path}")


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = run_dir / "config.json"
    config_data: dict = {}
    if config_path.exists():
        config_data = json.loads(config_path.read_text(encoding="utf-8"))

    jsonl_files = sorted(run_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {run_dir}")

    model_filter = set(args.models) if args.models else None
    pipeline_filter = set(args.pipelines) if args.pipelines else None

    records_by_col: dict[str, list[dict]] = {}
    questions_ordered: list[str] = []
    ground_truth_map: dict[str, str] = {}

    for jsonl_path in jsonl_files:
        records = load_jsonl(jsonl_path)
        if not records:
            continue

        model = records[0]["model"]
        pipeline = records[0]["pipeline"]

        if model_filter and model not in model_filter:
            continue
        if pipeline_filter and pipeline not in pipeline_filter:
            continue

        col_name = _column_name(model, pipeline)
        records_by_col[col_name] = records

        for r in records:
            if r["question"] not in ground_truth_map:
                questions_ordered.append(r["question"])
                ground_truth_map[r["question"]] = r.get("ground_truth", "")

    if not records_by_col:
        print("No matching JSONL files found after filtering.")
        return 1

    ground_truths = [ground_truth_map[q] for q in questions_ordered]

    print(f"Questions: {len(questions_ordered)}")
    print(f"Model+pipeline combinations: {len(records_by_col)}")
    print(f"Combinations: {', '.join(records_by_col.keys())}")

    responses_df = build_responses_sheet(records_by_col, questions_ordered)
    retrieval_df = build_retrieval_sheet(records_by_col, questions_ordered)
    golden_df = build_golden_answers_sheet(questions_ordered, ground_truths)
    run_info_df = build_run_info_sheet(run_dir, config_data)

    output_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "ragas_report.xlsx"
    )

    write_workbook(output_path, {
        "responses": responses_df,
        "retrieval": retrieval_df,
        "golden_answers": golden_df,
        "run_info": run_info_df,
    })

    print(
        f"\nNext step — run RAGAS evaluation:\n"
        f"  python evaluate_ragas_ollama.py \\\n"
        f"    --report {output_path} \\\n"
        f"    --responses {','.join(records_by_col.keys())}"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a baseline JSONL run folder into a RAGAS-compatible Excel workbook."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the timestamped baseline run folder (contains *.jsonl files).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output Excel path. Defaults to <run-dir>/ragas_report.xlsx.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Filter to specific models (e.g. qwen2.5:7b llama3.1:8b).",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=None,
        metavar="PIPELINE",
        help="Filter to specific pipelines (e.g. rag_only hybrid_rag).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
