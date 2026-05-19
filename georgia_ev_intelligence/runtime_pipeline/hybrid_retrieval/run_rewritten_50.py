"""Run the active hybrid retrieval pipeline over Rewritten_50_questions.xlsx."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from georgia_ev_intelligence.runtime_pipeline.generation.llm_client import generate_answer
from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext

from .factory import build_default_pipeline


PROMPT_TEMPLATE = """You are an analyst answering questions about an EV supply chain knowledge base
for the state of Georgia. Use ONLY the retrieved context below. Do not use
outside knowledge. Do not invent companies, roles, products, OEMs, locations,
employment numbers, or counts that are not present in the context.

Retrieved context:
{retrieved_parent_chunks}

User question:
{user_question}

---

Answer the question by following these style rules exactly.

1. OPENING LINE
   - If the question asks for a list, count, or set of matching items, begin
     with a one-sentence count statement that names what was found.
     Examples of the pattern (do not copy the contents — only the shape):
       "There are N <short restatement of the filter> in Georgia."
       "There is only 1 <short restatement of the filter> in Georgia."
   - If the question asks for a single fact (which company / which county /
     which location / highest / largest / lowest), open with the direct
     answer in one sentence, including the relevant attribute value in
     parentheses or after a colon. Do not add a count statement.
   - If no item in the context matches the question's filter, open with a
     sentence like:
       "There are no <restated filter> in Georgia."
       or
       "No <restated filter> are explicitly identified in the provided
       context."
     Then add one short sentence explaining the conclusion is based on the
     provided evidence. Do not speculate beyond the context.

2. BODY (when listing matching items)
   - One item per line. No bullets, no numbering, no markdown tables.
   - Format each line as:
       <Company Name> [<Tier>] | <FieldLabel>: <value> | <FieldLabel>: <value>
   - Include the tier in square brackets only when the question is about
     tiers or categories, or when the tier is informative for the answer.
   - Use a pipe character " | " to separate attributes on the same line.
   - The field labels should match the question's framing. Use these short
     labels when applicable: Role, Product, Produce, Employment, OEMs,
     Primary OEM, Primary OEMs, EV Supply Chain Role, Facility Type,
     EV Relevant, Industry Group, Updated Location, Address.
   - Only include attributes that the question asks for, or that the
     question's framing implies are relevant. Do not pad with extra fields.
   - Preserve company names, role values, location strings, and product
     descriptions exactly as they appear in the context (including
     capitalization, punctuation, ampersands, parentheses, and special
     characters).
   - Format employment as a plain integer. If a thousands separator helps
     readability for a single highlighted number in the opening line, use
     a comma; otherwise keep numbers bare.

3. GROUPING
   - If the question implies natural groups (for example, two different
     category values, or items split by tier), introduce each group with
     a short label line, then list its items beneath. Keep groups in the
     order the question presents them.

4. SCOPE AND GROUNDING
   - Every entity, attribute value, and count in your answer must be
     directly supported by the retrieved context. If a value is missing
     from the context for an item you would otherwise list, write "n/a"
     for that field rather than guessing.
   - If the question asks for a count, the count must equal the number of
     distinct items you actually list in the body.
   - If the context contains information that does not match the filter,
     ignore it. Do not mention non-matches.
   - Treat the same company appearing in multiple retrieved chunks as a
     single entity unless the question asks for per-location or per-site
     entries (in which case list each site as its own line).

5. TONE
   - Direct, factual, and concise. No preamble. No closing summary. No
     meta-commentary about the retrieval, the context, or your reasoning.
   - Do not mention "chunks", "retrieval", "the context", "the knowledge
     base", "the data", or how the answer was derived. The only acceptable
     reference is "based on the provided evidence" when explaining a
     no-result outcome.

Generate the answer now."""


OUTPUT_COLUMNS = [
    "s.no",
    "question",
    "golden answer",
    "human validated answer",
    "retrived context",
]


@dataclass(frozen=True)
class QuestionRow:
    serial_number: object
    question: str
    golden_answer: str


def main() -> int:
    args = _parse_args()
    input_path = _project_root() / "kb" / "Rewritten_50_questions.xlsx"
    output_path = args.output or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    questions = _load_questions(input_path=input_path, sheet_name=args.sheet)
    if args.limit is not None:
        questions = questions[: args.limit]

    retrieval_pipeline = build_default_pipeline()
    output_rows: list[dict[str, object]] = []

    for index, row in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {row.question}")
        retrieved_context = ""
        try:
            parent_contexts = retrieval_pipeline.retrieve(row.question)
            retrieved_context = _format_retrieved_context(parent_contexts)
            prompt = build_prompt(
                user_question=row.question,
                retrieved_parent_chunks=retrieved_context,
            )
        except Exception as exc:
            generated_answer = f"ERROR: retrieval failed: {exc}"
        else:
            try:
                generated_answer = generate_answer(prompt, timeout=args.llm_timeout)
            except Exception as exc:
                generated_answer = f"ERROR: LLM generation failed: {exc}"

        output_rows.append({
            "s.no": row.serial_number,
            "question": row.question,
            "golden answer": row.golden_answer,
            "human validated answer": generated_answer,
            "retrived context": retrieved_context,
        })
        _write_output(output_path, output_rows)

    print(f"Saved {len(output_rows)} rows to {output_path}")
    return 0


def build_prompt(user_question: str, retrieved_parent_chunks: str) -> str:
    """Build the exact final-answer prompt requested for this batch run."""
    return PROMPT_TEMPLATE.format(
        retrieved_parent_chunks=retrieved_parent_chunks,
        user_question=user_question,
    )


def _load_questions(input_path: Path, sheet_name: str) -> list[QuestionRow]:
    dataframe = pd.read_excel(input_path, sheet_name=sheet_name)
    _validate_columns(dataframe, input_path)

    rows: list[QuestionRow] = []
    for record in dataframe.to_dict(orient="records"):
        question = str(record["question"]).strip()
        if not question:
            continue

        rows.append(QuestionRow(
            serial_number=record["s.no"],
            question=question,
            golden_answer="" if pd.isna(record["answer"]) else str(record["answer"]),
        ))

    return rows


def _validate_columns(dataframe: pd.DataFrame, input_path: Path) -> None:
    required_columns = {"s.no", "question", "answer"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{input_path} is missing required columns: {missing_list}")


def _format_retrieved_context(parent_contexts: list[ParentContext]) -> str:
    return "\n\n".join(
        parent.parent_chunk_text
        for parent in parent_contexts
        if parent.parent_chunk_text
    )


def _write_output(output_path: Path, output_rows: list[dict[str, object]]) -> None:
    dataframe = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    dataframe.to_excel(output_path, index=False)


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        _project_root()
        / "georgia_ev_intelligence"
        / "outputs"
        / "hybrid_retrieval_rewritten_50"
        / f"{timestamp}_answers.xlsx"
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the active hybrid retrieval pipeline on "
            "kb/Rewritten_50_questions.xlsx and write generated answers to XLSX."
        )
    )
    parser.add_argument(
        "--sheet",
        default="Q&A",
        help="Worksheet name inside kb/Rewritten_50_questions.xlsx.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output XLSX path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke testing.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=180,
        help="Timeout in seconds for each LLM generation call.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
