"""Answer pipeline that uses the full Normalized_kb.xlsx as context without retrieval."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import pandas as pd

from georgia_ev_intelligence.shared import config


DIRECT_KB_PROMPT_TEMPLATE = """You are an analyst answering questions about the EV supply chain
for the state of Georgia. The full knowledge base is provided below as structured records.
Use ONLY this data. Do not use outside knowledge. Do not invent companies, roles, products,
OEMs, locations, employment numbers, or counts that are not present in the data.

Knowledge base records:
{kb_context}

User question:
{user_question}

---

Answer the question by following these style rules exactly.

1. OPENING LINE
   - If the question asks for a list, count, or set of matching items, begin with a
     one-sentence count statement.
   - If the question asks for a single fact, open with the direct answer in one sentence.
   - If no item matches the filter, open with "There are no <restated filter> in Georgia."

2. BODY (when listing items)
   - One item per line. No bullets, no numbering, no markdown tables.
   - Format: <Company Name> [<Tier>] | <FieldLabel>: <value> | <FieldLabel>: <value>
   - Only include fields the question asks for.
   - Preserve names, values, and special characters exactly as they appear.

3. SCOPE AND GROUNDING
   - Every entity and count must be directly supported by the knowledge base records.
   - If a value is missing for a listed item, write "n/a".

4. TONE
   - Direct, factual, concise. No preamble. No closing summary.

Generate the answer now."""


class DirectKBAnswerPipeline:
    """Answer questions using the full Normalized_kb.xlsx as context — no retrieval."""

    def __init__(
        self,
        answer_generator: Callable[[str, int], str],
        kb_path: Path | None = None,
    ) -> None:
        self._answer_generator = answer_generator
        self._kb_path = kb_path or config.GNEM_EXCEL
        self._kb_context: str | None = None
        self._kb_records: list[dict] | None = None

    def answer(
        self,
        question: str,
        timeout: int = 180,
    ) -> str:
        kb_context = self._load_kb_context()
        prompt = DIRECT_KB_PROMPT_TEMPLATE.format(
            kb_context=kb_context,
            user_question=question,
        )
        return self._answer_generator(prompt, timeout)

    def get_kb_records_as_text_list(self) -> list[str]:
        """Return each KB row as a formatted text string (used for RAGAS contexts)."""
        if self._kb_records is None:
            self._kb_records = _load_and_format_kb(self._kb_path)
        return self._kb_records

    def _load_kb_context(self) -> str:
        if self._kb_context is None:
            records = self.get_kb_records_as_text_list()
            self._kb_context = "\n".join(records)
        return self._kb_context


def _load_and_format_kb(kb_path: Path) -> list[str]:
    """Load Normalized_kb.xlsx and format each row as a compact text record."""
    df = pd.read_excel(kb_path)
    records: list[str] = []
    for _, row in df.iterrows():
        parts: list[str] = []
        for col in df.columns:
            if col == "_row_id":
                continue
            value = row[col]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            text = str(value).strip()
            if not text:
                continue
            parts.append(f"{col}: {text}")
        if parts:
            records.append(" | ".join(parts))
    return records
