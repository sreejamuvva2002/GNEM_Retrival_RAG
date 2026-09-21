"""Turn per-answer scores into the comparison tables you actually read."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MODES, MODE_LABELS

# Metrics summarised in the headline table, in reading order.
HEADLINE = [
    "judge_correctness",
    "judge_completeness",
    "judge_groundedness",
    "judge_relevance",
    "judge_overall",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "entity_recall",
    "entity_precision",
    "entity_f1",
    "numeric_recall",
    "token_f1",
    "rouge_l",
    "recall_at_k",
    "ndcg_at_k",
    "citation_rate",
    "is_refusal",
    "latency_ms",
    "input_tokens",
    "output_tokens",
]


def load_scores(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _mode_order(frame: pd.DataFrame) -> list[str]:
    return [mode for mode in MODES if mode in set(frame["mode"])]


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean per mode for every headline metric, with n and std."""
    primary = frame[frame.get("repeat", 0) == 0] if "repeat" in frame else frame
    records = []
    for mode in _mode_order(primary):
        subset = primary[primary["mode"] == mode]
        row: dict[str, object] = {"mode": mode, "description": MODE_LABELS.get(mode, mode), "n": len(subset)}
        for metric in HEADLINE:
            if metric not in subset:
                continue
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            row[metric] = round(float(values.mean()), 4) if len(values) else np.nan
            row[f"{metric}__sd"] = round(float(values.std(ddof=1)), 4) if len(values) > 1 else np.nan
        records.append(row)
    return pd.DataFrame(records)


def summarise_compact(frame: pd.DataFrame) -> pd.DataFrame:
    """The wide summary transposed to metric-per-row, "mean ± sd" per mode.

    The wide form is right for a spreadsheet; this is the one a human reads.
    """
    wide = summarise(frame)
    if not len(wide):
        return wide
    rows = []
    for metric in ["n"] + HEADLINE:
        if metric not in wide:
            continue
        row: dict[str, object] = {"metric": metric}
        for _, mode_row in wide.iterrows():
            value = mode_row.get(metric)
            sd = mode_row.get(f"{metric}__sd")
            if value is None or (isinstance(value, float) and np.isnan(value)):
                row[mode_row["mode"]] = "-"
            elif sd is None or (isinstance(sd, float) and np.isnan(sd)):
                row[mode_row["mode"]] = f"{value:g}"
            else:
                row[mode_row["mode"]] = f"{value:g} ± {sd:g}"
        rows.append(row)
    return pd.DataFrame(rows)


def by_category(frame: pd.DataFrame, metric: str = "judge_overall") -> pd.DataFrame:
    """Where each mode wins and loses - the table that drives routing decisions."""
    primary = frame[frame.get("repeat", 0) == 0] if "repeat" in frame else frame
    if metric not in primary or "category" not in primary:
        return pd.DataFrame()
    working = primary.copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    pivot = working.pivot_table(index="category", columns="mode", values=metric, aggfunc="mean")
    return pivot.round(3).reset_index()


def per_question(frame: pd.DataFrame, metric: str = "judge_overall") -> pd.DataFrame:
    primary = frame[frame.get("repeat", 0) == 0] if "repeat" in frame else frame
    if metric not in primary:
        return pd.DataFrame()
    working = primary.copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    pivot = working.pivot_table(index=["qid", "question"], columns="mode", values=metric, aggfunc="mean")
    modes = [mode for mode in MODES if mode in pivot.columns]
    if len(modes) >= 2:
        pivot["best_mode"] = pivot[modes].idxmax(axis=1)
        pivot["spread"] = (pivot[modes].max(axis=1) - pivot[modes].min(axis=1)).round(3)
    return pivot.round(3).reset_index()


def paired_significance(frame: pd.DataFrame, metric: str = "judge_overall") -> pd.DataFrame:
    """Paired comparison between modes on the same questions.

    n=30 is small, so a mean difference alone is not evidence. This runs a
    paired bootstrap (no normality assumption) and reports the 95% interval
    and win/loss/tie counts. If the interval spans zero, report the modes as
    indistinguishable on this metric rather than picking a winner.
    """
    primary = frame[frame.get("repeat", 0) == 0] if "repeat" in frame else frame
    if metric not in primary:
        return pd.DataFrame()
    working = primary.copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    pivot = working.pivot_table(index="qid", columns="mode", values=metric, aggfunc="mean")
    modes = [mode for mode in MODES if mode in pivot.columns]

    rows = []
    rng = np.random.default_rng(7)
    for i, left in enumerate(modes):
        for right in modes[i + 1 :]:
            paired = pivot[[left, right]].dropna()
            if paired.empty:
                continue
            diff = (paired[left] - paired[right]).to_numpy()
            samples = rng.choice(diff, size=(5000, len(diff)), replace=True).mean(axis=1)
            low, high = np.percentile(samples, [2.5, 97.5])
            rows.append(
                {
                    "metric": metric,
                    "mode_a": left,
                    "mode_b": right,
                    "n_paired": len(diff),
                    "mean_diff": round(float(diff.mean()), 4),
                    "ci95_low": round(float(low), 4),
                    "ci95_high": round(float(high), 4),
                    "significant": bool(low > 0 or high < 0),
                    "a_wins": int((diff > 0).sum()),
                    "b_wins": int((diff < 0).sum()),
                    "ties": int((diff == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def self_consistency(frame: pd.DataFrame) -> pd.DataFrame:
    """Stability across repeated runs of the same question (needs --repeats>1).

    Reported as the spread of the deterministic entity-F1 across repeats: a mode
    that swings between runs is not trustworthy even if its mean looks good.
    """
    if "repeat" not in frame or frame["repeat"].max() == 0:
        return pd.DataFrame()
    working = frame.copy()
    working["entity_f1"] = pd.to_numeric(working.get("entity_f1"), errors="coerce")
    grouped = working.groupby(["mode", "qid"])["entity_f1"].agg(["mean", "std", "count"]).reset_index()
    return (
        grouped.groupby("mode")
        .agg(mean_entity_f1=("mean", "mean"), mean_within_question_sd=("std", "mean"), repeats=("count", "mean"))
        .round(4)
        .reset_index()
    )


def cost_table(frame: pd.DataFrame, price_in: float, price_out: float) -> pd.DataFrame:
    if not price_in and not price_out:
        return pd.DataFrame()
    working = frame.copy()
    for column in ("input_tokens", "output_tokens"):
        working[column] = pd.to_numeric(working.get(column), errors="coerce").fillna(0)
    working["usd"] = working["input_tokens"] / 1e6 * price_in + working["output_tokens"] / 1e6 * price_out
    return (
        working.groupby("mode")
        .agg(
            total_usd=("usd", "sum"),
            usd_per_question=("usd", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
        )
        .round(6)
        .reset_index()
    )


def write_report(frame: pd.DataFrame, out_dir: Path, price_in: float = 0.0, price_out: float = 0.0) -> dict[str, Path]:
    """Write report.md plus an Excel workbook with one sheet per table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "summary": summarise_compact(frame),
        "summary_wide": summarise(frame),
        "by_category": by_category(frame),
        "per_question": per_question(frame),
        "significance": pd.concat(
            [paired_significance(frame, m) for m in ("judge_overall", "entity_f1", "faithfulness") if m in frame],
            ignore_index=True,
        )
        if len(frame)
        else pd.DataFrame(),
        "self_consistency": self_consistency(frame),
        "cost": cost_table(frame, price_in, price_out),
    }

    xlsx_path = out_dir / "report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, table in tables.items():
            (table if len(table) else pd.DataFrame({"note": ["no data"]})).to_excel(writer, sheet_name=name[:31], index=False)

    lines = ["# Evaluation report", ""]
    for name, table in tables.items():
        if not len(table) or name == "summary_wide":  # wide form lives in the workbook only
            continue
        lines.append(f"## {name.replace('_', ' ').title()}")
        lines.append("")
        lines.append(table.to_markdown(index=False))
        lines.append("")
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines))

    return {"markdown": md_path, "excel": xlsx_path}
