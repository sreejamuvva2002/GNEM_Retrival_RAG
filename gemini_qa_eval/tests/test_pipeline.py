"""End-to-end coverage in mock mode: no API key, no network, still exercises
every stage the real run uses."""

import json

import pandas as pd

from gqe.cli import main
from gqe.config import MODES, Settings
from gqe.data import load_company_docs, load_questions
from gqe.evaluation import aggregate
from gqe.gemini_client import GeminiClient
from gqe.manual import agreement as agreement_mod
from gqe.manual import workbook as workbook_mod
from gqe.modes import answer_question
from gqe.retrieval import Retriever


def test_questions_load_with_reference_answers():
    questions = load_questions(limit=30)
    assert len(questions) == 30
    assert all(q.question for q in questions)
    assert sum(q.has_reference for q in questions) == 30


def test_company_workbook_loads_as_documents():
    docs = load_company_docs()
    assert len(docs) > 100
    assert all(doc.doc_id and doc.text for doc in docs)


def test_each_mode_produces_an_answer_and_the_right_evidence():
    cfg = Settings()
    client = GeminiClient(cfg)
    question = load_questions(limit=1)[0]
    retriever = Retriever(load_company_docs(), client=client, cfg=cfg)

    records = {mode: answer_question(question, mode, client=client, retriever=retriever, cfg=cfg) for mode in MODES}

    assert records["llm_only"].contexts == []          # no retrieval in mode 1
    assert records["llm_kb"].contexts                   # retrieval in modes 2 and 3
    assert records["llm_kb_web"].contexts
    assert all(record.answer for record in records.values())


def test_full_cli_run(tmp_path, monkeypatch):
    monkeypatch.setenv("GQE_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("GQE_CACHE_DIR", str(tmp_path / "cache"))

    assert main(["all", "--limit", "2", "--run-id", "t", "--reviewers", "a,b"]) == 0
    run_dir = tmp_path / "t"

    answers = [json.loads(line) for line in (run_dir / "answers.jsonl").read_text().splitlines()]
    assert len(answers) == 2 * len(MODES)

    scored = [json.loads(line) for line in (run_dir / "scored.jsonl").read_text().splitlines()]
    assert all("judge_overall" in row and "token_f1" in row for row in scored)
    assert any("faithfulness" in row for row in scored)

    assert (run_dir / "report.md").exists()
    assert (run_dir / "report.xlsx").exists()
    assert (run_dir / "manual_review.xlsx").exists()


def test_report_tables_cover_every_mode(tmp_path):
    rows = [json.loads(line) for line in open("runs/smoke/scored.jsonl")] if _smoke_exists() else None
    if rows is None:
        return
    frame = pd.DataFrame(rows)
    summary = aggregate.summarise(frame)
    assert set(summary["mode"]) == set(MODES)
    assert len(aggregate.paired_significance(frame)) == 3   # three mode pairs
    assert len(aggregate.self_consistency(frame)) == len(MODES)


def _smoke_exists() -> bool:
    from pathlib import Path

    return Path("runs/smoke/scored.jsonl").exists()


def test_manual_workbook_is_blinded_and_round_trips(tmp_path):
    scored = [
        {
            "qid": f"Q00{i}",
            "mode": mode,
            "repeat": 0,
            "question": f"question {i}",
            "category": "cat",
            "golden_answer": "reference",
            "answer": f"answer {mode}",
            "context_titles": ["Alpha"],
            "web_sources": [],
        }
        for i in (1, 2, 3)
        for mode in MODES
    ]
    path = workbook_mod.build_workbook(scored, tmp_path / "review.xlsx", reviewers=["a", "b"], overlap_fraction=0.3)

    review = pd.read_excel(path, sheet_name="review")
    assert "mode" not in review.columns                      # the mode must not leak to reviewers
    assert set(review["blind_label"]) <= set(workbook_mod.BLIND_LABELS)
    assert {"correctness", "completeness", "groundedness", "relevance", "notes"} <= set(review.columns)

    restored = workbook_mod.load_completed(path)
    assert set(restored["mode"].dropna()) == set(MODES)      # and it must come back on load


def test_agreement_tables_from_a_filled_workbook(tmp_path):
    scored_rows = []
    human_rows = []
    for i in range(1, 7):
        for mode, score in zip(MODES, (3, 4, 5)):
            qid = f"Q00{i}"
            scored_rows.append(
                {"qid": qid, "mode": mode, "repeat": 0, "judge_correctness": score, "judge_completeness": score,
                 "judge_groundedness": score, "judge_relevance": score}
            )
            human_rows.append(
                {"row_id": f"{qid}-{mode}", "qid": qid, "mode": mode, "correctness": score, "completeness": score,
                 "groundedness": score, "relevance": score, "usable_as_is": "yes", "hallucination": "no",
                 "reference_disputed": "no"}
            )
    human = pd.DataFrame(human_rows)
    scores = pd.DataFrame(scored_rows)

    comparison = agreement_mod.human_vs_judge(human, scores)
    assert set(comparison["dimension"]) == set(agreement_mod.DIMENSIONS)
    assert comparison["quadratic_kappa"].min() == 1.0        # identical scores => perfect agreement

    ranking = agreement_mod.mode_ranking(human, scores)
    assert ranking["rank_agrees"].all()

    path = agreement_mod.write_agreement_report(human, scores, tmp_path)
    assert path.exists()
