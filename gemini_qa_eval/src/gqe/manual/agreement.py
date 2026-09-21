"""Does the automatic scoring agree with the humans?

This is the step that decides whether you are allowed to trust the automatic
numbers for the next run. Three questions:

1. **Do reviewers agree with each other?** (on the duplicated overlap rows)
   Quadratic-weighted Cohen's kappa. If humans do not agree with humans, the
   rubric is ambiguous - fix the rubric before blaming the judge.
2. **Does the judge agree with the humans?** Same kappa, plus Spearman
   correlation and exact/within-1 agreement rates, per dimension.
3. **Does the judge rank the modes the same way humans do?** The rank
   correlation over mode means. A judge can disagree on absolute scores and
   still be perfectly usable if it orders the systems correctly.

Rules of thumb for the kappa: <0.4 do not use the judge unsupervised;
0.4-0.6 usable for tracking regressions, not for go/no-go decisions;
>0.6 comparable to a second human reviewer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DIMENSIONS = ["correctness", "completeness", "groundedness", "relevance"]


def quadratic_kappa(a: list[float], b: list[float], min_rating: int = 1, max_rating: int = 5) -> float:
    """Cohen's kappa with quadratic weights - a 5-vs-4 disagreement costs far
    less than 5-vs-1, which is the right behaviour for ordinal rubric scores."""
    pairs = [(int(round(x)), int(round(y))) for x, y in zip(a, b) if _ok(x) and _ok(y)]
    if len(pairs) < 2:
        return float("nan")
    n = max_rating - min_rating + 1
    observed = np.zeros((n, n))
    for x, y in pairs:
        observed[x - min_rating, y - min_rating] += 1
    observed /= observed.sum()
    rows = observed.sum(axis=1)
    cols = observed.sum(axis=0)
    expected = np.outer(rows, cols)
    weights = np.array([[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)])
    denominator = (weights * expected).sum()
    if denominator == 0:
        return float("nan")
    return float(1 - (weights * observed).sum() / denominator)


def _ok(value) -> bool:
    try:
        return not pd.isna(value) and float(value) > 0
    except (TypeError, ValueError):
        return False


def human_vs_judge(human: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Per-dimension agreement between the human scores and the judge scores."""
    merged = human.merge(
        scores[["qid", "mode"] + [f"judge_{d}" for d in DIMENSIONS if f"judge_{d}" in scores]],
        on=["qid", "mode"],
        how="inner",
    )
    rows = []
    for dimension in DIMENSIONS:
        judge_column = f"judge_{dimension}"
        if dimension not in merged or judge_column not in merged:
            continue
        pair = merged[[dimension, judge_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if pair.empty:
            continue
        human_scores = pair[dimension].to_numpy()
        judge_scores = pair[judge_column].to_numpy()
        rows.append(
            {
                "dimension": dimension,
                "n": len(pair),
                "human_mean": round(float(human_scores.mean()), 3),
                "judge_mean": round(float(judge_scores.mean()), 3),
                "bias_judge_minus_human": round(float(judge_scores.mean() - human_scores.mean()), 3),
                "quadratic_kappa": round(quadratic_kappa(human_scores, judge_scores), 3),
                "spearman": round(_spearman(human_scores, judge_scores), 3),
                "exact_agreement": round(float((np.round(human_scores) == np.round(judge_scores)).mean()), 3),
                "within_1": round(float((np.abs(human_scores - judge_scores) <= 1).mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def inter_annotator(human: pd.DataFrame) -> pd.DataFrame:
    """Agreement between reviewers, measured on the duplicated overlap rows."""
    working = human.copy()
    working["base_row"] = working["row_id"].astype(str).str.replace("-dup", "", regex=False)
    duplicated = working[working.duplicated("base_row", keep=False)]
    rows = []
    for dimension in DIMENSIONS:
        if dimension not in duplicated:
            continue
        first, second = [], []
        for _, group in duplicated.groupby("base_row"):
            values = pd.to_numeric(group[dimension], errors="coerce").dropna().tolist()
            if len(values) >= 2:
                first.append(values[0])
                second.append(values[1])
        if len(first) >= 2:
            rows.append(
                {
                    "dimension": dimension,
                    "n_overlap_rows": len(first),
                    "quadratic_kappa": round(quadratic_kappa(first, second), 3),
                    "within_1": round(float(np.mean(np.abs(np.array(first) - np.array(second)) <= 1)), 3),
                }
            )
    return pd.DataFrame(rows)


def mode_ranking(human: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Do humans and the judge order the three modes the same way?"""
    human_means = human.copy()
    for dimension in DIMENSIONS:
        if dimension in human_means:
            human_means[dimension] = pd.to_numeric(human_means[dimension], errors="coerce")
    available = [d for d in DIMENSIONS if d in human_means]
    if not available or "mode" not in human_means:
        return pd.DataFrame()
    human_mean = human_means.groupby("mode")[available].mean().mean(axis=1)

    judge_columns = [f"judge_{d}" for d in available if f"judge_{d}" in scores]
    if not judge_columns:
        return pd.DataFrame()
    judge_frame = scores.copy()
    for column in judge_columns:
        judge_frame[column] = pd.to_numeric(judge_frame[column], errors="coerce")
    judge_mean = judge_frame.groupby("mode")[judge_columns].mean().mean(axis=1)

    table = pd.DataFrame({"human_mean": human_mean, "judge_mean": judge_mean}).dropna()
    if table.empty:
        return table.reset_index()
    table["human_rank"] = table["human_mean"].rank(ascending=False)
    table["judge_rank"] = table["judge_mean"].rank(ascending=False)
    table["rank_agrees"] = table["human_rank"] == table["judge_rank"]
    return table.round(3).reset_index()


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        value = spearmanr(a, b).statistic
    except ImportError:
        rank_a = pd.Series(a).rank().to_numpy()
        rank_b = pd.Series(b).rank().to_numpy()
        if rank_a.std() == 0 or rank_b.std() == 0:
            return float("nan")
        value = float(np.corrcoef(rank_a, rank_b)[0, 1])
    return float(value) if value == value else float("nan")


def human_summary(human: pd.DataFrame) -> pd.DataFrame:
    """What the humans said, per mode - the ground truth of the whole exercise."""
    working = human.copy()
    for dimension in DIMENSIONS:
        if dimension in working:
            working[dimension] = pd.to_numeric(working[dimension], errors="coerce")
    available = [d for d in DIMENSIONS if d in working]
    grouped = working.groupby("mode")[available].mean().round(3)
    if "usable_as_is" in working:
        usable = working.groupby("mode")["usable_as_is"].apply(
            lambda values: float((values.astype(str).str.lower() == "yes").mean())
        )
        grouped["usable_as_is_rate"] = usable.round(3)
    if "hallucination" in working:
        hallucinated = working.groupby("mode")["hallucination"].apply(
            lambda values: float((values.astype(str).str.lower() == "yes").mean())
        )
        grouped["hallucination_rate"] = hallucinated.round(3)
    if "reference_disputed" in working:
        disputed = working.groupby("mode")["reference_disputed"].apply(
            lambda values: float((values.astype(str).str.lower() == "yes").mean())
        )
        grouped["reference_disputed_rate"] = disputed.round(3)
    return grouped.reset_index()


def write_agreement_report(human: pd.DataFrame, scores: pd.DataFrame, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "human_summary": human_summary(human),
        "human_vs_judge": human_vs_judge(human, scores),
        "inter_annotator": inter_annotator(human),
        "mode_ranking": mode_ranking(human, scores),
    }
    path = out_dir / "agreement.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in tables.items():
            (table if len(table) else pd.DataFrame({"note": ["no data"]})).to_excel(writer, sheet_name=name[:31], index=False)
    (out_dir / "agreement.json").write_text(
        json.dumps({name: table.to_dict("records") for name, table in tables.items()}, indent=2, default=str)
    )
    return path
