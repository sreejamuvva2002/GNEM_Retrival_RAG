"""LLM-based query rewriter for structured entity extraction.

Calls the configured Ollama model to parse a raw user question into
a StructuredQuery. Degrades gracefully on any failure by returning
an empty StructuredQuery with rewrite_successful=False.
"""
from __future__ import annotations

import json
import logging
import time

import requests

from ...shared import config
from .models import StructuredQuery

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a structured query parser for a Georgia EV supply chain \
knowledge base. Extract search filters from the user question.
Respond ONLY with valid JSON. No explanation. No markdown.
No preamble. No trailing text.

The JSON must have exactly these keys (all are lists of strings,
use empty list [] if not mentioned):
{
  "locations": [],
  "product_services": [],
  "companies": [],
  "oems": [],
  "ev_supply_chain_roles": [],
  "industry_groups": [],
  "facility_types": []
}

Rules:
- Normalise to lowercase.
- Split multi-value fields (e.g. "battery cells and EV motors" \
-> ["battery cells", "ev motors"]).
- Do not infer -- only extract what the user explicitly mentions.
- Location means city, county, region, or state in Georgia.
- If the question is too vague to extract filters, return all \
empty lists."""


class QueryRewriter:
    """Uses the configured Ollama model to parse a raw user question
    into a StructuredQuery.

    Respects QUERY_REWRITER_ENABLED flag.
    Falls back to empty StructuredQuery on any failure.
    """

    def rewrite(self, question: str) -> StructuredQuery:
        """Parse a question into a StructuredQuery.

        If config.QUERY_REWRITER_ENABLED is False, returns _fallback()
        immediately. Otherwise tries _call_llm() up to
        config.MAX_REWRITER_RETRIES times. On any exception, logs a
        warning and returns _fallback().
        """
        if not config.QUERY_REWRITER_ENABLED:
            return self._fallback(question)

        start = time.time()
        last_error: Exception | None = None

        for attempt in range(config.MAX_REWRITER_RETRIES):
            try:
                raw = self._call_llm(question)
                latency_ms = (time.time() - start) * 1000
                return self._parse_response(raw, question, latency_ms)
            except Exception as e:
                last_error = e
                logger.warning(
                    "Query rewrite attempt %d failed: %s", attempt + 1, e
                )

        latency_ms = (time.time() - start) * 1000
        logger.warning(
            "Query rewrite exhausted retries, falling back. Last error: %s",
            last_error,
        )
        return self._fallback(question, latency_ms)

    def _call_llm(self, question: str) -> dict:
        """POST to Ollama /api/generate and return parsed JSON dict.

        Raises ValueError if response is not valid JSON.
        """
        prompt = self._build_prompt(question)
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.QUERY_REWRITER_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 256},
            },
            timeout=config.QUERY_REWRITER_TIMEOUT,
        )
        resp.raise_for_status()

        raw_text = resp.json().get("response", "").strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw_text = "\n".join(lines).strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {raw_text[:200]}") from e

    def _build_prompt(self, question: str) -> str:
        """Build the full prompt string using system + user template."""
        return f"{_SYSTEM_PROMPT}\n\nUSER:\n{question}"

    def _parse_response(
        self, raw: dict, question: str, latency_ms: float
    ) -> StructuredQuery:
        """Parse the dict returned by the LLM into a StructuredQuery.

        For each key, takes only string values from the list,
        strips and lowercases each value, and filters empty strings.
        Sets rewrite_successful=True.
        """

        def _clean_list(key: str) -> list[str]:
            values = raw.get(key, [])
            if not isinstance(values, list):
                return []
            cleaned = []
            for v in values:
                if isinstance(v, str):
                    s = v.strip().lower()
                    if s:
                        cleaned.append(s)
            return cleaned

        return StructuredQuery(
            original_question=question,
            locations=_clean_list("locations"),
            product_services=_clean_list("product_services"),
            companies=_clean_list("companies"),
            oems=_clean_list("oems"),
            ev_supply_chain_roles=_clean_list("ev_supply_chain_roles"),
            industry_groups=_clean_list("industry_groups"),
            facility_types=_clean_list("facility_types"),
            rewrite_successful=True,
            rewrite_latency_ms=latency_ms,
        )

    def _fallback(self, question: str, latency_ms: float = 0.0) -> StructuredQuery:
        """Return a StructuredQuery with all empty lists and rewrite_successful=False.

        Used when rewriter is disabled or fails.
        """
        return StructuredQuery(
            original_question=question,
            rewrite_successful=False,
            rewrite_latency_ms=latency_ms,
        )
