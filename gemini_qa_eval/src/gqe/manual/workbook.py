"""Build the blinded human-review workbook.

Design choices that keep the human scores usable:

* **Blinded.** The mode is replaced by a per-question shuffled label
  (System X / Y / Z). Reviewers who know which one is "the RAG one" score it
  higher; the mapping lives on a separate sheet nobody opens while scoring.
* **Shuffled row order.** Stops reviewers anchoring on the first system.
* **Same rubric as the judge.** Identical dimensions and anchors, so human and
  judge scores are directly comparable and kappa means something.
* **Dropdowns, not free text,** for the scores - typed "4/5" and "four" ruin
  the join. Free text belongs in the notes column.
* **Overlap rows.** A slice of rows is assigned to two reviewers so
  inter-annotator agreement can be measured; without it you cannot tell a
  disagreeing judge from a noisy rubric.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

BLIND_LABELS = ["System X", "System Y", "System Z"]

RUBRIC_ROWS = [
    ("correctness", "Do the facts agree with the reference answer?", "5 = all facts agree | 3 = core right, a supporting fact wrong | 1 = contradicts the reference"),
    ("completeness", "How much of the reference answer's content is covered?", "5 = all key entities/counts | 3 = about half | 1 = almost none"),
    ("groundedness", "Are claims supported by the evidence shown to the model?", "5 = every claim traceable | 3 = no evidence was supplied to this system | 1 = contradicts the evidence"),
    ("relevance", "Does it answer the question asked, without padding?", "5 = exactly on target | 3 = answers but padded | 1 = evasive or off-topic"),
    ("usable_as_is", "Would you send this to a stakeholder without editing?", "yes / minor_edits / no"),
    ("hallucination", "Any claim that is confidently stated and not supportable?", "yes / no - if yes, quote it in notes"),
]

INSTRUCTIONS = [
    "HOW TO REVIEW",
    "",
    "1. Work on the 'review' sheet, top to bottom. Do not sort or filter it.",
    "2. For each row: read the question, then the reference answer, then the system answer.",
    "3. Score the four 1-5 dimensions using the dropdowns. Use the anchors on the 'rubric' sheet.",
    "4. Fill 'usable_as_is' and 'hallucination'. If hallucination = yes, quote the claim in 'notes'.",
    "5. If the reference answer itself looks wrong, set reference_disputed = yes and explain in notes.",
    "   (This is how errors in the golden answers get found - do not silently score around them.)",
    "6. Leave a row blank if you genuinely cannot judge it. A blank is better than a guess.",
    "",
    "RULES",
    "- System names are blinded on purpose. Do not try to work out which is which.",
    "- Length is not quality. A short correct answer beats a long hedged one.",
    "- Judge only against the reference and the evidence shown, not your own memory.",
    "- Budget roughly 3-5 minutes per row. Take a break every 20 rows; fatigue shows up as drift.",
    "",
    "WHEN YOU ARE DONE",
    "Save the file and hand it back unchanged in structure (same sheet and column names),",
    "then run:  python -m gqe.cli agreement --workbook <this file> --scores <scored.jsonl>",
]


def build_workbook(
    scored_rows: list[dict],
    out_path: Path,
    *,
    seed: int = 7,
    reviewers: list[str] | None = None,
    overlap_fraction: float = 0.2,
    include_evidence: bool = True,
) -> Path:
    """Write the review workbook. ``scored_rows`` is the scored JSONL, loaded."""
    rng = random.Random(seed)
    reviewers = reviewers or ["reviewer_1"]

    primary = [row for row in scored_rows if int(row.get("repeat", 0)) == 0]
    by_question: dict[str, list[dict]] = {}
    for row in primary:
        by_question.setdefault(row["qid"], []).append(row)

    records = []
    for qid in sorted(by_question):
        group = by_question[qid]
        labels = BLIND_LABELS[: len(group)]
        rng.shuffle(labels)
        for label, row in zip(labels, group):
            evidence = ""
            if include_evidence:
                titles = row.get("context_titles") or []
                web = [source.get("uri", "") for source in (row.get("web_sources") or [])]
                parts = []
                if titles:
                    parts.append("Internal records shown: " + ", ".join(titles[:15]))
                if web:
                    parts.append("Web sources used: " + ", ".join(web[:8]))
                evidence = "\n".join(parts) or "No evidence supplied to this system."
            records.append(
                {
                    "row_id": f"{qid}-{label.replace(' ', '')}",
                    "qid": qid,
                    "blind_label": label,
                    "category": row.get("category", ""),
                    "question": row.get("question", ""),
                    "reference_answer": row.get("golden_answer", ""),
                    "system_answer": row.get("answer", ""),
                    "evidence_shown": evidence,
                    "correctness": "",
                    "completeness": "",
                    "groundedness": "",
                    "relevance": "",
                    "usable_as_is": "",
                    "hallucination": "",
                    "reference_disputed": "",
                    "notes": "",
                    "reviewer": "",
                    "_mode": row.get("mode", ""),
                }
            )

    rng.shuffle(records)

    # Assign reviewers round-robin, then duplicate a slice for agreement measurement.
    for index, record in enumerate(records):
        record["reviewer"] = reviewers[index % len(reviewers)]
    if len(reviewers) > 1 and overlap_fraction > 0:
        overlap_n = max(1, int(len(records) * overlap_fraction))
        for record in rng.sample(records, overlap_n):
            duplicate = dict(record)
            others = [r for r in reviewers if r != record["reviewer"]]
            duplicate["reviewer"] = rng.choice(others)
            duplicate["row_id"] = record["row_id"] + "-dup"
            records.append(duplicate)

    frame = pd.DataFrame(records)
    key = frame[["row_id", "qid", "blind_label", "_mode"]].rename(columns={"_mode": "mode"})
    review = frame.drop(columns=["_mode"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame({"instructions": INSTRUCTIONS}).to_excel(writer, sheet_name="instructions", index=False)
        pd.DataFrame(RUBRIC_ROWS, columns=["dimension", "question to ask", "anchors"]).to_excel(
            writer, sheet_name="rubric", index=False
        )
        review.to_excel(writer, sheet_name="review", index=False)
        key.to_excel(writer, sheet_name="key_do_not_open", index=False)
        _decorate(writer.book["review"], len(review))

    return out_path


def _decorate(sheet, n_rows: int) -> None:
    """Dropdowns, widths, wrapping - the difference between a workbook people
    fill in and one they give up on."""
    from openpyxl.styles import Alignment, Font

    headers = [cell.value for cell in sheet[1]]
    widths = {
        "row_id": 14, "qid": 8, "blind_label": 12, "category": 24, "question": 50,
        "reference_answer": 60, "system_answer": 60, "evidence_shown": 40, "notes": 40, "reviewer": 14,
    }
    for index, header in enumerate(headers, start=1):
        letter = get_column_letter(index)
        sheet.column_dimensions[letter].width = widths.get(header, 15)
        sheet.cell(row=1, column=index).font = Font(bold=True)

    for header in ("question", "reference_answer", "system_answer", "evidence_shown", "notes"):
        if header not in headers:
            continue
        letter = get_column_letter(headers.index(header) + 1)
        for row in range(2, n_rows + 2):
            sheet[f"{letter}{row}"].alignment = Alignment(wrap_text=True, vertical="top")

    validations = {
        ("correctness", "completeness", "groundedness", "relevance"): '"1,2,3,4,5"',
        ("usable_as_is",): '"yes,minor_edits,no"',
        ("hallucination", "reference_disputed"): '"yes,no"',
    }
    for columns, formula in validations.items():
        validation = DataValidation(type="list", formula1=formula, allow_blank=True, showDropDown=True)
        sheet.add_data_validation(validation)
        for header in columns:
            if header not in headers:
                continue
            letter = get_column_letter(headers.index(header) + 1)
            validation.add(f"{letter}2:{letter}{n_rows + 1}")

    sheet.freeze_panes = "A2"


def load_completed(path: Path) -> pd.DataFrame:
    """Read a filled-in workbook and re-attach the blinded mode mapping."""
    review = pd.read_excel(path, sheet_name="review")
    key = pd.read_excel(path, sheet_name="key_do_not_open")
    key["row_id"] = key["row_id"].astype(str)
    review["row_id"] = review["row_id"].astype(str)
    base = key.set_index("row_id")["mode"].to_dict()
    review["mode"] = review["row_id"].map(lambda r: base.get(r) or base.get(r.replace("-dup", "")))
    return review


def dump_rubric_json(path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps({dimension: {"asks": asks, "anchors": anchors} for dimension, asks, anchors in RUBRIC_ROWS}, indent=2))
    return path
