"""Estimate the token budget for a full run, before spending anything.

Counts the real prompts this project will send - same system prompts, same
retrieved context, same question set - and reports input/output tokens per
stage, per mode, and in total.

    PYTHONPATH=src python scripts/estimate_cost.py
    PYTHONPATH=src python scripts/estimate_cost.py --repeats 3 --judge-samples 3
    PYTHONPATH=src python scripts/estimate_cost.py --price-in 0.30 --price-out 2.50

Token counts are estimated at CHARS_PER_TOKEN characters per token rather than
by calling the tokenizer, so the run can be costed with no API key. That is
accurate to roughly +/-15% for English prose; treat the output as a budget, not
an invoice. Once you have a real run, `report.xlsx` -> `cost` reports the
tokens actually billed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gqe.config import MODE_KB, MODE_KB_WEB, MODE_LLM, Settings  # noqa: E402
from gqe.data import load_company_docs, load_questions  # noqa: E402
from gqe.evaluation.judge import RUBRIC  # noqa: E402
from gqe.modes import SYSTEM_KB, SYSTEM_KB_WEB, SYSTEM_LLM_ONLY, _user_prompt  # noqa: E402
from gqe.retrieval import Retriever, format_context  # noqa: E402

CHARS_PER_TOKEN = 4.0

# Fixed prompt scaffolding in the evaluation layers, measured in characters.
CLAIMS_INSTRUCTION = 240
VERDICT_INSTRUCTION = 300
RELEVANCY_INSTRUCTION = 180
USEFULNESS_INSTRUCTION = 260
RECALL_INSTRUCTION = 260
JUDGE_SYSTEM_CHARS = 260


def tokens(chars: float) -> float:
    return chars / CHARS_PER_TOKEN


@dataclass
class Stage:
    name: str
    calls: int = 0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    embed_tokens: float = 0.0
    note: str = ""

    def add(self, *, input_chars: float = 0, output_chars: float = 0, embed_chars: float = 0, calls: int = 1) -> None:
        self.calls += calls
        self.input_tokens += tokens(input_chars)
        self.output_tokens += tokens(output_chars)
        self.embed_tokens += tokens(embed_chars)


@dataclass
class Budget:
    stages: dict[str, Stage] = field(default_factory=dict)

    def stage(self, name: str, note: str = "") -> Stage:
        if name not in self.stages:
            self.stages[name] = Stage(name, note=note)
        return self.stages[name]


def build(args) -> tuple[Budget, dict]:
    cfg = Settings()
    questions = load_questions(limit=args.limit or cfg.n_questions, cfg=cfg)
    docs = load_company_docs(cfg=cfg)
    retriever = Retriever(docs, cfg=cfg)

    modes = [MODE_LLM, MODE_KB, MODE_KB_WEB] if args.modes == "all" else args.modes.split(",")
    budget = Budget()

    # Document embeddings: once per corpus, then cached forever.
    if cfg.retrieval_strategy == "hybrid" and any(m != MODE_LLM for m in modes):
        embed_stage = budget.stage("embeddings: corpus", "one-off, cached across all future runs")
        embed_stage.add(embed_chars=sum(len(doc.text) for doc in docs), calls=len(docs))

    systems = {MODE_LLM: SYSTEM_LLM_ONLY, MODE_KB: SYSTEM_KB, MODE_KB_WEB: SYSTEM_KB_WEB}
    context_chars: dict[str, int] = {}
    answer_chars: dict[str, float] = {}

    for question in questions:
        context = ""
        if any(mode != MODE_LLM for mode in modes):
            context = format_context(retriever.search(question.question, cfg.top_k))
            context_chars[question.qid] = len(context)

        # Output size proxy: these questions ask for company lists, so the golden
        # answer is the best available predictor of how long the answer will be.
        golden = len(question.golden_answer) or 900
        grounded_answer = min(golden * args.answer_ratio, cfg.max_output_tokens * CHARS_PER_TOKEN)
        answer_chars[question.qid] = grounded_answer

        for mode in modes:
            stage = budget.stage(f"answers: {mode}")
            prompt = _user_prompt(question.question, context if mode != MODE_LLM else None)
            out = grounded_answer * (0.6 if mode == MODE_LLM else 1.0)  # ungrounded answers run shorter
            for _ in range(args.repeats):
                stage.add(input_chars=len(systems[mode]) + len(prompt), output_chars=out)

            if cfg.retrieval_strategy == "hybrid" and mode != MODE_LLM:
                budget.stage("embeddings: queries").add(embed_chars=len(question.question), calls=1)

    if args.no_score:
        return budget, {"questions": len(questions), "modes": modes, "docs": len(docs)}

    # --- evaluation layers -------------------------------------------------
    for question in questions:
        context = context_chars.get(question.qid, 0)
        answer = answer_chars[question.qid]
        reference = len(question.golden_answer)

        for mode in modes:
            grounded = mode != MODE_LLM
            ctx = context if grounded else 0

            # Layer 3: judge. One call per answer, per sample.
            judge = budget.stage("judge: absolute grading")
            for _ in range(args.judge_samples):
                judge.add(
                    input_chars=JUDGE_SYSTEM_CHARS + len(RUBRIC) + len(question.question) + reference + ctx + answer,
                    output_chars=700,  # evidence + 4 scores + rationale, JSON
                )

            if not grounded:
                continue

            # Layer 4: faithfulness = claim extraction + claim verification.
            faith = budget.stage("rag: faithfulness")
            faith.add(input_chars=CLAIMS_INSTRUCTION + answer, output_chars=answer * 1.1)
            faith.add(input_chars=VERDICT_INSTRUCTION + ctx + answer * 1.1, output_chars=answer * 1.4)

            # Layer 4: answer relevancy = 3 reverse-generated questions + embeddings.
            relevancy = budget.stage("rag: answer relevancy")
            relevancy.add(input_chars=RELEVANCY_INSTRUCTION + answer, output_chars=400)
            budget.stage("embeddings: relevancy").add(embed_chars=len(question.question) + 400, calls=4)

            # Layer 4: context precision and recall.
            budget.stage("rag: context precision").add(
                input_chars=USEFULNESS_INSTRUCTION + len(question.question) + reference + ctx, output_chars=200
            )
            budget.stage("rag: context recall").add(
                input_chars=RECALL_INSTRUCTION + ctx + reference, output_chars=200
            )

    if args.pairwise:
        pairs = len(modes) * (len(modes) - 1) // 2
        stage = budget.stage("judge: pairwise")
        for question in questions:
            answer = answer_chars[question.qid]
            for _ in range(pairs):
                stage.add(
                    input_chars=JUDGE_SYSTEM_CHARS + 500 + len(question.question) + len(question.golden_answer) + answer * 2,
                    output_chars=400,
                )

    meta = {
        "questions": len(questions),
        "modes": modes,
        "docs": len(docs),
        "mean_context_chars": int(sum(context_chars.values()) / len(context_chars)) if context_chars else 0,
        "top_k": cfg.top_k,
    }
    return budget, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate the token budget for a full run")
    parser.add_argument("--limit", type=int, default=None, help="number of questions (default N_QUESTIONS)")
    parser.add_argument("--modes", default="all")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--judge-samples", type=int, default=1)
    parser.add_argument("--pairwise", action="store_true", help="include the pairwise preference pass")
    parser.add_argument("--no-score", action="store_true", help="answers only, no evaluation layers")
    parser.add_argument("--answer-ratio", type=float, default=1.2, help="answer length as a multiple of the golden answer")
    parser.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens")
    parser.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens")
    parser.add_argument("--price-embed", type=float, default=0.0, help="USD per 1M embedding tokens")
    args = parser.parse_args()

    budget, meta = build(args)

    print(f"{meta['questions']} questions x {len(meta['modes'])} modes over {meta['docs']} company records")
    if meta.get("mean_context_chars"):
        print(f"mean retrieved context: {meta['mean_context_chars']:,} chars (~{int(tokens(meta['mean_context_chars'])):,} tokens) at top_k={meta['top_k']}")
    print()

    header = f"{'stage':<28}{'calls':>8}{'input tok':>14}{'output tok':>13}{'embed tok':>12}"
    print(header)
    print("-" * len(header))
    totals = [0, 0.0, 0.0, 0.0]
    for stage in budget.stages.values():
        print(f"{stage.name:<28}{stage.calls:>8,}{stage.input_tokens:>14,.0f}{stage.output_tokens:>13,.0f}{stage.embed_tokens:>12,.0f}")
        totals[0] += stage.calls
        totals[1] += stage.input_tokens
        totals[2] += stage.output_tokens
        totals[3] += stage.embed_tokens
    print("-" * len(header))
    print(f"{'TOTAL':<28}{totals[0]:>8,}{totals[1]:>14,.0f}{totals[2]:>13,.0f}{totals[3]:>12,.0f}")
    print()
    print(f"Buy at least: {totals[1]:,.0f} input + {totals[2]:,.0f} output tokens"
          + (f" + {totals[3]:,.0f} embedding tokens" if totals[3] else ""))
    print(f"With 30% headroom for retries and re-runs: {totals[1] * 1.3:,.0f} input + {totals[2] * 1.3:,.0f} output")

    if args.price_in or args.price_out:
        cost = totals[1] / 1e6 * args.price_in + totals[2] / 1e6 * args.price_out + totals[3] / 1e6 * args.price_embed
        print(f"\nAt the prices given: ${cost:,.2f} for one full run (${cost * 1.3:,.2f} with headroom)")
        print("Mode 3's Google Search grounding is billed per request, separately, and is NOT in this figure.")

    print("\nEstimated at ~4 chars/token, no tokenizer call. Budget, not invoice: expect +/-15%.")
    print("Every call is cached on disk, so re-running after a crash or a code change re-bills only what changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
