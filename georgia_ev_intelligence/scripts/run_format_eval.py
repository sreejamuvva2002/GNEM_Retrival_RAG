"""
scripts/run_format_eval.py
─────────────────────────────────────────────────────────────────────────────
Run the 4-format evaluation across models and generate comparison data.

Usage:
  # Run a single format + model:
  venv\\Scripts\\python scripts\\run_format_eval.py --format 1 --model qwen2.5:7b

  # Run all formats for one model:
  venv\\Scripts\\python scripts\\run_format_eval.py --format all --model qwen2.5:7b

  # Run specific format for multiple models:
  venv\\Scripts\\python scripts\\run_format_eval.py --format 3 --model qwen2.5:7b gemma2:9b

  # Generate dashboard from existing results:
  venv\\Scripts\\python scripts\\run_format_eval.py --dashboard-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.logger import get_logger
from shared.config import Config
from evaluate.format_runner import (
    run_format1, run_format2, run_format3, run_format4,
    check_few_shot_contamination, _call_llm,
)

logger = get_logger("run_format_eval")

_OUTPUT_DIR = Path("outputs/format_eval")
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load the 50 evaluation questions ─────────────────────────────────────────

def load_eval_questions() -> list[dict]:
    """
    Load evaluation questions from questions_export.json.

    The export uses keys: 'Question', 'Num', 'Use Case Category', 'Human validated answers'
    Normalize to lowercase keys expected by the rest of the pipeline:
      'question', 'id', 'category', 'golden'
    """
    q_path = Path("outputs/questions_export.json")
    if q_path.exists():
        with open(q_path, encoding="utf-8") as f:
            raw = json.load(f)

        normalized = []
        for i, item in enumerate(raw, 1):
            # Handle both old lowercase format and new export format
            q_text = (
                item.get("question")          # old format
                or item.get("Question")       # export format (capital Q)
                or item.get("q", "")
            )
            if not q_text:
                continue
            normalized.append({
                "id":       item.get("id") or item.get("Num") or f"Q{i}",
                "category": item.get("category") or item.get("Use Case Category", "GENERAL"),
                "question": q_text.strip(),
                "golden":   item.get("golden") or item.get("Human validated answers", ""),
            })
        return normalized

    # Fallback: import directly from run_ragas_eval.py
    try:
        from scripts.run_ragas_eval import SMOKE_QUESTIONS
        return SMOKE_QUESTIONS
    except Exception:
        raise FileNotFoundError(
            "No questions found. Run: venv\\Scripts\\python scripts\\run_ragas_eval.py --questions 50"
        )


# ── LLM-as-Judge scorer ───────────────────────────────────────────────────────

_JUDGE_PROMPT = """You are an expert evaluator for a Georgia EV Supply Chain Q&A system.

Score the following answer on a scale of 0.0 to 1.0 for each metric.
CRITICAL: Use precise decimal values (e.g., 0.84, 0.91, 0.65) instead of rounding to quartiles (do NOT just use 0.25, 0.5, 0.75, 1.0).
Return ONLY a valid JSON object, nothing else.

QUESTION: {question}

RETRIEVED CONTEXT (what the system found in the database):
{context}

SYSTEM ANSWER: {answer}

GOLDEN ANSWER (the ground truth):
{golden}

Score these 5 metrics (0.00 to 1.00):
1. faithfulness: Does the answer use only information from the retrieved context? (1.00 = purely faithful. If context is empty, this MUST be 0.00 because nothing can be faithful to an empty context).
2. answer_relevancy: Does the answer actually address the question? (1.00 = perfectly relevant)
3. context_precision: Are the retrieved rows relevant to the question? (1.00 = all relevant. If context is empty, this MUST be 0.00).
4. context_recall: Did the retrieval capture all the facts present in the GOLDEN ANSWER? (1.00 = complete. If context is empty, this MUST be 0.00).
5. answer_correctness: Is the SYSTEM ANSWER factually correct compared to the GOLDEN ANSWER? (1.00 = identical facts)

Return exactly this JSON format (use your own precise scores):
{{"faithfulness": 0.84, "answer_relevancy": 0.92, "context_precision": 0.76, "context_recall": 0.88, "answer_correctness": 0.95}}"""


def score_answer(question: str, context: str, answer: str, golden: str, judge_model: str = "qwen2.5:7b") -> dict:
    """Score an answer using LLM-as-judge. Returns dict of metric scores."""
    prompt = _JUDGE_PROMPT.format(
        question=question[:300],
        context=context[:1500] if context else "(No retrieved context — Format 2 No-RAG evaluation)",
        answer=answer[:800],
        golden=golden[:800] if golden else "No golden answer provided.",
    )
    raw = _call_llm(prompt, max_tokens=200, model=judge_model)

    # Parse JSON response
    try:
        # Find JSON object in response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            scores = json.loads(raw[start:end])
            # Validate all expected keys exist
            for key in ["faithfulness", "answer_relevancy", "context_precision",
                        "context_recall", "answer_correctness"]:
                if key not in scores:
                    scores[key] = None
            return scores
    except Exception:
        pass

    # Fallback if parsing fails
    return {
        "faithfulness": None, "answer_relevancy": None,
        "context_precision": None, "context_recall": None,
        "answer_correctness": None, "parse_error": raw[:100],
    }


# ── Run evaluation for a single format + model ────────────────────────────────

def run_evaluation(
    format_num: int,
    model: str,
    questions: list[dict],
    judge_model: str = "qwen2.5:7b",
) -> list[dict]:
    """Run all questions through one format and score each answer."""
    print(f"\n{'='*65}")
    print(f"  FORMAT {format_num} | Model: {model} | {len(questions)} questions")
    print(f"{'='*65}\n")

    # Load pipeline only for formats that need it
    # WHY: Config uses @lru_cache — os.environ changes after first import are ignored.
    # We pass the model string directly to EVAgent and _call_llm instead.
    pipeline = None
    if format_num in (1, 3, 4):
        from phase4_agent.pipeline import EVAgent
        pipeline = EVAgent(model_override=model)   # model passed directly

    results = []
    total_scores = {
        "faithfulness": [], "answer_relevancy": [], "context_precision": [],
        "context_recall": [], "answer_correctness": [],
    }

    for i, q_item in enumerate(questions, 1):
        question = q_item["question"]
        q_id = q_item.get("id", f"Q{i}")
        category = q_item.get("category", "GENERAL")

        print(f"  [{i:02d}/{len(questions)}] {q_id} [{category}]: {question[:60]}...")
        t0 = time.monotonic()

        try:
            # Run the appropriate format
            if format_num == 1:
                result = run_format1(question, pipeline, model=model)
            elif format_num == 2:
                result = run_format2(question, model=model)
            elif format_num == 3:
                result = run_format3(question, pipeline)
            elif format_num == 4:
                result = run_format4(question, pipeline, model=model)
            else:
                raise ValueError(f"Unknown format: {format_num}")

            # Score the answer
            scores = score_answer(
                question=question,
                context=result.get("retrieved_context", ""),
                answer=result.get("answer", ""),
                golden=q_item.get("golden", ""),
                judge_model=judge_model,
            )

        except Exception as exc:
            logger.error("Q%s failed: %s", q_id, exc)
            result = {"format": f"F{format_num}", "question": question,
                      "answer": f"[ERROR: {exc}]", "retrieved_context": "",
                      "retrieved_count": 0, "elapsed_s": 0, "few_shot_used": False}
            scores = {"faithfulness": 0, "answer_relevancy": 0,
                      "context_precision": None, "context_recall": None,
                      "answer_correctness": 0}

        elapsed = time.monotonic() - t0

        # Accumulate scores (skip nulls)
        for metric, val in scores.items():
            if val is not None and metric in total_scores:
                total_scores[metric].append(float(val))

        row = {
            "id": q_id,
            "category": category,
            "question": question,
            "answer": result.get("answer", ""),
            "retrieved_context": result.get("retrieved_context", ""),
            "retrieved_count": result.get("retrieved_count", 0),
            "few_shot_used": result.get("few_shot_used", False),
            "elapsed_s": result.get("elapsed_s", round(elapsed, 1)),
            "format": f"F{format_num}",
            "model": model,
            "scores": scores,
            "golden": q_item.get("golden", ""),
        }
        results.append(row)

        # Print progress
        fa = scores.get("faithfulness", "-")
        ar = scores.get("answer_relevancy", "-")
        ac = scores.get("answer_correctness", "-")
        fa_str = f"{fa:.2f}" if isinstance(fa, float) else str(fa)
        ar_str = f"{ar:.2f}" if isinstance(ar, float) else str(ar)
        ac_str = f"{ac:.2f}" if isinstance(ac, float) else str(ac)
        print(f"       faith={fa_str} relevancy={ar_str} correctness={ac_str} | {elapsed:.1f}s")

    # Print summary
    print(f"\n  {'─'*50}")
    print(f"  FORMAT {format_num} | {model} — SUMMARY")
    for metric, vals in total_scores.items():
        if vals:
            avg = sum(vals) / len(vals)
            print(f"    {metric:<22}: {avg:.3f} ({len(vals)} scored)")
    print(f"  {'─'*50}\n")

    return results


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(results: list[dict], format_num: int, model: str) -> Path:
    """Save results to JSONL file."""
    model_slug = model.replace(":", "_").replace(".", "_")
    out_path = _OUTPUT_DIR / f"f{format_num}_{model_slug}_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved: {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Georgia EV Multi-Format Evaluator")
    parser.add_argument(
        "--format", type=str, default="3",
        help="Format number: 1|2|3|4|all (default: 3)"
    )
    parser.add_argument(
        "--model", nargs="+", default=["qwen2.5:7b"],
        help="Model(s) to use (space-separated)"
    )
    parser.add_argument(
        "--questions", type=int, default=50,
        help="Number of questions to evaluate (default: 50)"
    )
    parser.add_argument(
        "--judge-model", default="qwen2.5:7b",
        help="Model to use as LLM judge (default: qwen2.5:7b)"
    )
    parser.add_argument(
        "--dashboard-only", action="store_true",
        help="Skip evaluation, just regenerate HTML dashboard from existing results"
    )
    parser.add_argument(
        "--check-contamination", action="store_true",
        help="Check few-shot/eval question overlap before running"
    )
    args = parser.parse_args()

    if args.dashboard_only:
        print("Generating dashboard from existing results...")
        from scripts.generate_dashboard import generate_dashboard
        generate_dashboard()
        return

    # Load questions
    questions = load_eval_questions()[:args.questions]
    print(f"\nLoaded {len(questions)} evaluation questions")

    # Contamination check
    if args.check_contamination:
        print("\nChecking few-shot contamination...")
        report = check_few_shot_contamination([q["question"] for q in questions])
        print(json.dumps(report, indent=2))
        if report.get("contamination_rate", 0) > 0.1:
            print("\n⚠️  High contamination — results may be biased. Proceed? [y/n]")
            if input().lower() != "y":
                return

    # Determine formats to run
    if args.format == "all":
        formats = [1, 2, 3, 4]
    else:
        formats = [int(f) for f in args.format.split(",")]

    all_result_paths = list(_OUTPUT_DIR.glob("f*_results.jsonl"))

    for model in args.model:
        print(f"\n{'#'*65}")
        print(f"  MODEL: {model}")
        print(f"{'#'*65}")

        for fmt in formats:
            results = run_evaluation(
                format_num=fmt,
                model=model,
                questions=questions,
                judge_model=args.judge_model,
            )
            path = save_results(results, fmt, model)
            all_result_paths.append(path)

    # Auto-generate dashboard
    print("\nGenerating HTML comparison dashboard...")
    try:
        from scripts.generate_dashboard import generate_dashboard
        generate_dashboard()
    except Exception as exc:
        logger.warning("Dashboard generation failed: %s — run manually", exc)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
