"""
Runs all 50 human-validated questions through the pipeline and scores results.
Scoring: token-overlap F1 between generated answer and human-validated answer.
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path

import pandas as pd

from .. import config, pipeline


def load_qa() -> list[dict]:
    df = pd.read_excel(config.HUMAN_QA_EXCEL)
    cols_lower = {c.lower(): c for c in df.columns}

    num_col = next((c for k, c in cols_lower.items() if k in ("num", "number", "#", "id")), df.columns[0])
    cat_col = next((c for k, c in cols_lower.items() if "category" in k or "use case" in k), None)
    q_col   = next((c for k, c in cols_lower.items() if "question" in k), df.columns[2])
    a_col   = next((c for k, c in cols_lower.items() if "human" in k or "validated" in k), df.columns[3])

    records = []
    for _, row in df.iterrows():
        q = str(row[q_col]).strip()
        a = str(row[a_col]).strip()
        if not q or q == "nan":
            continue
        records.append({
            "num": row[num_col],
            "category": str(row[cat_col]).strip() if cat_col else "",
            "question": q,
            "human_answer": a,
        })
    return records


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in re.split(r"\W+", text) if len(w) >= 3}


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    tp = len(pred_tokens & gold_tokens)
    precision = tp / len(pred_tokens)
    recall    = tp / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_all(verbose: bool = True) -> list[dict]:
    qa_pairs = load_qa()
    results = []

    for item in qa_pairs:
        num = item["num"]
        question = item["question"]
        human_answer = item["human_answer"]

        if verbose:
            print(f"[Q{num:02d}] {question[:80]}...")

        t0 = time.time()
        try:
            pr = pipeline.run(question)
            elapsed = time.time() - t0
            f1 = token_f1(pr.answer, human_answer)

            record = {
                "num": num,
                "category": item["category"],
                "question": question,
                "human_answer": human_answer,
                "generated_answer": pr.answer,
                "f1_score": round(f1, 4),
                "support_level": pr.support_level,
                "hallucination_risk": pr.hallucination_risk,
                "retrieval_method": pr.retrieval_method,
                "evidence_count": pr.evidence_count,
                "intent": pr.intent.get("type", ""),
                "filters_applied": pr.filters_applied,
                "elapsed_s": round(elapsed, 2),
            }
        except Exception as exc:
            elapsed = time.time() - t0
            record = {
                "num": num,
                "category": item["category"],
                "question": question,
                "human_answer": human_answer,
                "generated_answer": f"ERROR: {exc}",
                "f1_score": 0.0,
                "support_level": "Error",
                "hallucination_risk": "High",
                "retrieval_method": "error",
                "evidence_count": 0,
                "intent": "",
                "filters_applied": {},
                "elapsed_s": round(elapsed, 2),
            }

        results.append(record)

        if verbose:
            print(f"       F1={record['f1_score']:.2f}  support={record['support_level']}  "
                  f"evidence={record['evidence_count']}  risk={record['hallucination_risk']}  "
                  f"({record['elapsed_s']}s)\n")

    return results


def save_results(results: list[dict]) -> Path:
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    out = config.OUTPUTS_DIR / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return out


def print_summary(results: list[dict]) -> None:
    f1_scores = [r["f1_score"] for r in results]
    avg_f1 = sum(f1_scores) / len(f1_scores)
    support_counts: dict[str, int] = {}
    for r in results:
        support_counts[r["support_level"]] = support_counts.get(r["support_level"], 0) + 1

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY  ({len(results)} questions)")
    print("=" * 60)
    print(f"Average F1 Score : {avg_f1:.3f}")
    print(f"F1 ≥ 0.80        : {sum(1 for s in f1_scores if s >= 0.80)} questions")
    print(f"F1 ≥ 0.50        : {sum(1 for s in f1_scores if s >= 0.50)} questions")
    print(f"F1 = 0.00        : {sum(1 for s in f1_scores if s == 0.0)} questions")
    print("\nSupport Level Distribution:")
    for level, count in sorted(support_counts.items(), key=lambda x: -x[1]):
        print(f"  {level:<35} {count:>3}")
    print("=" * 60)
