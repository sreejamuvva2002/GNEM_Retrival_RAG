"""Command line entry point.

    python -m gqe.cli answer     --modes all --repeats 1
    python -m gqe.cli score      --answers runs/<id>/answers.jsonl
    python -m gqe.cli report     --scores  runs/<id>/scored.jsonl
    python -m gqe.cli workbook   --scores  runs/<id>/scored.jsonl
    python -m gqe.cli agreement  --workbook reviewed.xlsx --scores runs/<id>/scored.jsonl
    python -m gqe.cli all                       # answer -> score -> report -> workbook
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from .config import MODES, Settings
from .data import company_names, load_company_docs, load_questions
from .evaluation import aggregate, deterministic, judge, rag_metrics, retrieval_metrics
from .gemini_client import GeminiClient
from .manual import agreement as agreement_mod
from .manual import workbook as workbook_mod
from .modes import answer_question
from .retrieval import Retriever

log = logging.getLogger("gqe")


def _run_dir(cfg: Settings, run_id: str | None) -> Path:
    path = Path(cfg.runs_dir) / (run_id or time.strftime("%Y%m%d-%H%M%S"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_jsonl(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _latest_run(cfg: Settings, filename: str) -> Path:
    candidates = sorted(Path(cfg.runs_dir).glob(f"*/{filename}"))
    if not candidates:
        raise SystemExit(f"no {filename} found under {cfg.runs_dir}; run the previous step first")
    return candidates[-1]


# ---------------------------------------------------------------------------
# answer
# ---------------------------------------------------------------------------
def cmd_answer(args, cfg: Settings) -> Path:
    client = GeminiClient(cfg)
    questions = load_questions(limit=args.limit or cfg.n_questions, cfg=cfg)
    modes = list(MODES) if args.modes == "all" else args.modes.split(",")
    docs = load_company_docs(cfg=cfg) if any(mode != "llm_only" for mode in modes) else []
    retriever = Retriever(docs, client=client, cfg=cfg) if docs else None

    out_dir = _run_dir(cfg, args.run_id)
    rows: list[dict] = []
    total = len(questions) * len(modes) * args.repeats
    done = 0
    for question in questions:
        for mode in modes:
            for repeat in range(args.repeats):
                record = answer_question(
                    question,
                    mode,
                    client=client,
                    retriever=retriever,
                    repeat=repeat,
                    temperature=cfg.temperature if repeat == 0 else 0.7,
                    cfg=cfg,
                )
                rows.append(record.to_dict())
                done += 1
                status = "ERROR" if record.error else f"{len(record.answer)} chars"
                log.info("[%d/%d] %s %s -> %s", done, total, question.qid, mode, status)

    path = _write_jsonl(rows, out_dir / "answers.jsonl")
    _write_jsonl(
        [{"config": {k: str(v) for k, v in vars(cfg).items() if k != "api_key"}, "modes": modes, "n_questions": len(questions)}],
        out_dir / "run_meta.jsonl",
    )
    print(f"wrote {len(rows)} answers -> {path}")
    if cfg.mock:
        print("NOTE: mock mode - no live Gemini calls were made. Set GEMINI_API_KEY for real answers.")
    return path


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def cmd_score(args, cfg: Settings) -> Path:
    answers_path = Path(args.answers) if args.answers else _latest_run(cfg, "answers.jsonl")
    rows = _read_jsonl(answers_path)
    docs = load_company_docs(cfg=cfg)
    entities = company_names(docs)
    client = GeminiClient(cfg)

    scored: list[dict] = []
    for index, row in enumerate(rows, start=1):
        answer = row.get("answer", "")
        reference = row.get("golden_answer", "")
        contexts = row.get("contexts", []) or []
        record = dict(row)

        # Layer 1 - deterministic (always).
        record.update(deterministic.score_answer(answer, reference, entities))

        # Layer 2 - retrieval (grounded modes only).
        if contexts:
            gold_ids = retrieval_metrics.gold_doc_ids(reference, docs)
            record.update(retrieval_metrics.retrieval_scores(row.get("context_ids", []), gold_ids, cfg.top_k))

        # Layers 3 and 4 - model-graded (skippable; they cost tokens).
        if not args.no_judge:
            record.update(
                judge.grade_answer(
                    question=row.get("question", ""),
                    answer=answer,
                    reference=reference,
                    context="\n".join(contexts),
                    client=client,
                    n_samples=args.judge_samples,
                    cfg=cfg,
                )
            )
        if not args.no_rag_metrics and contexts:
            record.update(
                rag_metrics.score_rag(
                    question=row.get("question", ""),
                    answer=answer,
                    reference=reference,
                    contexts=contexts,
                    client=client,
                    cfg=cfg,
                )
            )
        scored.append(record)
        log.info("[%d/%d] scored %s %s", index, len(rows), row.get("qid"), row.get("mode"))

    path = _write_jsonl(scored, answers_path.parent / "scored.jsonl")
    print(f"wrote {len(scored)} scored rows -> {path}")
    return path


# ---------------------------------------------------------------------------
# pairwise
# ---------------------------------------------------------------------------
def cmd_pairwise(args, cfg: Settings) -> Path:
    scores_path = Path(args.scores) if args.scores else _latest_run(cfg, "scored.jsonl")
    rows = [row for row in _read_jsonl(scores_path) if int(row.get("repeat", 0)) == 0]
    client = GeminiClient(cfg)

    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(row["qid"], {})[row["mode"]] = row

    comparisons = []
    for qid, by_mode in sorted(grouped.items()):
        modes = [mode for mode in MODES if mode in by_mode]
        for i, left in enumerate(modes):
            for right in modes[i + 1 :]:
                result = judge.compare_answers(
                    question=by_mode[left]["question"],
                    reference=by_mode[left].get("golden_answer", ""),
                    answer_a=by_mode[left].get("answer", ""),
                    answer_b=by_mode[right].get("answer", ""),
                    label_a=left,
                    label_b=right,
                    client=client,
                    cfg=cfg,
                )
                result["qid"] = qid
                comparisons.append(result)
                log.info("%s %s vs %s -> %s", qid, left, right, result["winner"])

    path = _write_jsonl(comparisons, scores_path.parent / "pairwise.jsonl")
    frame = pd.DataFrame(comparisons)
    if len(frame):
        tally = frame.groupby(["mode_a", "mode_b", "winner"]).size().reset_index(name="n")
        tally.to_csv(scores_path.parent / "pairwise_tally.csv", index=False)
        print(tally.to_string(index=False))
    print(f"wrote {len(comparisons)} pairwise judgements -> {path}")
    return path


# ---------------------------------------------------------------------------
# report / workbook / agreement
# ---------------------------------------------------------------------------
def cmd_report(args, cfg: Settings) -> Path:
    scores_path = Path(args.scores) if args.scores else _latest_run(cfg, "scored.jsonl")
    frame = aggregate.load_scores(scores_path)
    paths = aggregate.write_report(frame, scores_path.parent, cfg.price_input_per_1m, cfg.price_output_per_1m)
    print(aggregate.summarise_compact(frame).to_string(index=False))
    print(f"\nreport -> {paths['markdown']}\n         {paths['excel']}")
    return paths["markdown"]


def cmd_workbook(args, cfg: Settings) -> Path:
    scores_path = Path(args.scores) if args.scores else _latest_run(cfg, "scored.jsonl")
    rows = _read_jsonl(scores_path)
    reviewers = args.reviewers.split(",") if args.reviewers else ["reviewer_1"]
    path = workbook_mod.build_workbook(
        rows,
        scores_path.parent / "manual_review.xlsx",
        reviewers=reviewers,
        overlap_fraction=args.overlap,
    )
    print(f"blinded review workbook -> {path}")
    print("Send it to the reviewers; they fill the 'review' sheet and send it back.")
    return path


def cmd_agreement(args, cfg: Settings) -> Path:
    scores_path = Path(args.scores) if args.scores else _latest_run(cfg, "scored.jsonl")
    human = workbook_mod.load_completed(Path(args.workbook))
    scores = aggregate.load_scores(scores_path)
    path = agreement_mod.write_agreement_report(human, scores, scores_path.parent)
    print(agreement_mod.human_summary(human).to_string(index=False))
    print()
    print(agreement_mod.human_vs_judge(human, scores).to_string(index=False))
    print(f"\nagreement report -> {path}")
    return path


def cmd_all(args, cfg: Settings) -> None:
    answers = cmd_answer(args, cfg)
    args.answers = str(answers)
    scored = cmd_score(args, cfg)
    args.scores = str(scored)
    cmd_report(args, cfg)
    cmd_workbook(args, cfg)


def build_parser() -> argparse.ArgumentParser:
    # -v is accepted both before and after the subcommand; SUPPRESS keeps the
    # subparser from overwriting a flag given at the top level.
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(prog="gqe", description="Gemini three-mode QA benchmark", parents=[base])
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str):
        return subparsers.add_parser(name, help=help_text, parents=[base])

    def common(sub):
        sub.add_argument("--run-id", default=None, help="reuse a run directory name")
        sub.add_argument("--limit", type=int, default=None, help="number of questions (default N_QUESTIONS)")
        sub.add_argument("--modes", default="all", help="all | llm_only,llm_kb,llm_kb_web")
        sub.add_argument("--repeats", type=int, default=1, help=">1 enables the self-consistency metric")
        return sub

    def scoring(sub):
        sub.add_argument("--answers", default=None)
        sub.add_argument("--scores", default=None)
        sub.add_argument("--no-judge", action="store_true", help="skip the LLM judge (no judge tokens)")
        sub.add_argument("--no-rag-metrics", action="store_true", help="skip faithfulness / relevancy / context metrics")
        sub.add_argument("--judge-samples", type=int, default=1, help=">1 takes the median and reports spread")
        return sub

    common(add("answer", "generate answers for all three modes")).set_defaults(func=cmd_answer)
    scoring(add("score", "score answers (layers 1-4)")).set_defaults(func=cmd_score)

    pairwise = add("pairwise", "blind pairwise preference between modes")
    pairwise.add_argument("--scores", default=None)
    pairwise.set_defaults(func=cmd_pairwise)

    report = add("report", "build report.md / report.xlsx")
    report.add_argument("--scores", default=None)
    report.set_defaults(func=cmd_report)

    book = add("workbook", "build the blinded human-review workbook")
    book.add_argument("--scores", default=None)
    book.add_argument("--reviewers", default=None, help="comma-separated names; 2+ enables inter-annotator agreement")
    book.add_argument("--overlap", type=float, default=0.2, help="fraction of rows double-reviewed")
    book.set_defaults(func=cmd_workbook)

    agree = add("agreement", "compare human scores with the judge")
    agree.add_argument("--workbook", required=True)
    agree.add_argument("--scores", default=None)
    agree.set_defaults(func=cmd_agreement)

    everything = scoring(common(add("all", "answer -> score -> report -> workbook")))
    everything.add_argument("--reviewers", default=None)
    everything.add_argument("--overlap", type=float, default=0.2)
    everything.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = Settings()
    args.func(args, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
