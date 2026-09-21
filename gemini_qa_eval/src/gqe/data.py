"""Loading the two inputs: the question set and the internal company workbook."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings, settings as default_settings

# Column names differ between the workbooks people hand over, so match loosely.
_QUESTION_COLS = ("question", "questions", "query", "prompt")
_ANSWER_COLS = ("human validated answers", "golden answer", "answer", "expected answer", "ground truth")
_CATEGORY_COLS = ("use case category", "category", "type", "theme")


@dataclass
class Question:
    qid: str
    question: str
    golden_answer: str = ""
    category: str = "uncategorised"

    @property
    def has_reference(self) -> bool:
        return bool(self.golden_answer.strip())


@dataclass
class Doc:
    """One row of the company workbook, rendered for retrieval and prompting."""

    doc_id: str
    title: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for candidate in candidates:  # fall back to substring match
        for lower, original in lowered.items():
            if candidate in lower:
                return original
    return None


def load_questions(path: Path | None = None, limit: int | None = None, cfg: Settings | None = None) -> list[Question]:
    """Read the question set from .xlsx or .csv.

    Any sheet works as long as it has a question column; a golden-answer column
    turns on the reference-based metrics, and a category column turns on the
    per-category breakdown in the report.
    """
    cfg = cfg or default_settings
    path = Path(path or cfg.questions_path)
    limit = cfg.n_questions if limit is None else limit

    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    columns = list(frame.columns)
    q_col = _pick(columns, _QUESTION_COLS)
    if q_col is None:
        raise ValueError(f"{path} has no question column; found {columns}")
    a_col = _pick(columns, _ANSWER_COLS)
    c_col = _pick(columns, _CATEGORY_COLS)

    questions: list[Question] = []
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        text = str(row[q_col]).strip()
        if not text or text.lower() == "nan":
            continue
        questions.append(
            Question(
                qid=f"Q{position:03d}",
                question=text,
                golden_answer="" if a_col is None else str(row[a_col] or "").strip().replace("nan", ""),
                category="uncategorised" if c_col is None else str(row[c_col] or "uncategorised").strip(),
            )
        )
        if limit and len(questions) >= limit:
            break
    return questions


def load_company_docs(path: Path | None = None, cfg: Settings | None = None) -> list[Doc]:
    """Turn every row of the internal workbook into one retrievable record."""
    cfg = cfg or default_settings
    path = Path(path or cfg.company_data_path)
    frame = pd.read_excel(path)
    frame = frame.where(pd.notna(frame), None)

    name_col = _pick(list(frame.columns), ("company", "company name", "name", "entity"))
    docs: list[Doc] = []
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        fields = {str(k).strip(): v for k, v in row.items() if v not in (None, "")}
        if not fields:
            continue
        title = str(fields.get(name_col, f"Record {position}")) if name_col else f"Record {position}"
        body = "\n".join(f"{key}: {value}" for key, value in fields.items())
        docs.append(Doc(doc_id=f"D{position:04d}", title=title, text=f"{title}\n{body}", meta=fields))
    return docs


def company_names(docs: list[Doc]) -> list[str]:
    """Distinct company names, longest first, for entity matching in scoring."""
    names = {doc.title.strip() for doc in docs if len(doc.title.strip()) > 2}
    return sorted(names, key=len, reverse=True)


_PUNCT = re.compile(r"[^a-z0-9\s]")


def normalise_name(name: str) -> str:
    """Company-name key that tolerates the usual suffix noise (Inc., LLC, Co.)."""
    text = _PUNCT.sub(" ", str(name).lower())
    tokens = [t for t in text.split() if t not in {"inc", "llc", "co", "corp", "ltd", "company", "the", "lp", "plc", "gmbh"}]
    return " ".join(tokens).strip()
