"""
Phase 1 — Entity Extractor
Extracts structured facts from raw text using Ollama (qwen2.5:14b).
Stores investment amounts, job counts, locations → PostgreSQL gev_extracted_facts.

Why structured extraction matters:
  When someone asks "How much did Hanwha invest in Georgia?"
  → SQL query: SELECT fact_value_numeric FROM gev_extracted_facts
               WHERE company_name='Hanwha Q Cells' AND fact_type='investment'
  → Returns in <200ms instead of searching 1000 embeddings.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

import httpx

from shared.config import Config
from shared.db import ExtractedFact, get_session
from shared.logger import get_logger

logger = get_logger("phase1.entity_extractor")

# Prompt for Ollama — extract structured facts from document text
_EXTRACTION_PROMPT = """\
You are a structured data extractor for the Georgia EV supply chain.
Extract ALL key facts from the document text below.

For each fact found, output a JSON array where each element has:
- fact_type: one of [investment, jobs_created, jobs_total, facility_expansion, plant_capacity, oem_partnership, product_launch, policy_incentive, other]
- fact_value_text: exact text of the fact (e.g. "$400 million")
- fact_value_numeric: numeric value only (e.g. 400000000 for $400M, 3000 for 3,000 jobs)
- fact_currency: "USD" if monetary, null otherwise
- fact_unit: "USD" / "jobs" / "sqft" / "MW" / "GWh" / "vehicles" / null
- fact_year: year as integer if mentioned (e.g. 2024), null if not clear
- fact_quarter: "Q1"/"Q2"/"Q3"/"Q4" if mentioned, null otherwise
- location_city: city if mentioned, null otherwise
- location_county: county in Georgia if mentioned, null otherwise
- oem_partner: OEM company mentioned (Kia, Hyundai, Ford, etc.), null if none
- confidence_score: 0.0-1.0 (how confident you are in this extraction)
- source_sentence: the exact sentence this fact was extracted from

Rules:
- Only extract facts explicitly stated in the text
- Do not invent numbers not in the text
- If a value uses "million" or "billion", convert: 1M = 1000000, 1B = 1000000000
- Return ONLY a valid JSON array, no other text
- If no facts found, return []

Company context: {company_name}
Document text (first 3000 chars):
{text}

JSON array:"""


def _call_ollama_sync(prompt: str, model: str, base_url: str) -> str:
    """Synchronous Ollama call for entity extraction."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,         # Disable chain-of-thought for faster JSON output
        "format": "json",       # Force JSON output mode
        "options": {
            "temperature": 0.0,  # Fully deterministic
            "num_predict": 1024,
            "num_ctx": 4096,
        },
    }
    response = httpx.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def _parse_facts_json(raw: str) -> list[dict[str, Any]]:
    """
    Parse LLM output as JSON array of facts.
    Handles common LLM quirks: markdown fences, trailing commas, etc.
    """
    if not raw or not raw.strip():
        return []

    # Strip markdown fences if present
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    # If the model returned a JSON object with a list inside, unwrap it
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            # Look for any list value
            for v in obj.values():
                if isinstance(v, list):
                    return v
        except json.JSONDecodeError:
            pass

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        # Try to extract partial JSON array
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    logger.debug("Could not parse facts JSON from LLM output: %s", raw[:200])
    return []


def _coerce_fact(raw: dict[str, Any], company_id: int | None, company_name: str, document_id: int | None) -> dict[str, Any]:
    """Coerce a raw extracted fact dict into DB model fields."""
    numeric = raw.get("fact_value_numeric")
    if numeric is not None:
        try:
            numeric = float(str(numeric).replace(",", ""))
        except (ValueError, TypeError):
            numeric = None

    year = raw.get("fact_year")
    if year is not None:
        try:
            year = int(year)
            if year < 1990 or year > 2035:
                year = None
        except (ValueError, TypeError):
            year = None

    confidence = raw.get("confidence_score", 0.5)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.5

    return {
        "document_id": document_id,
        "company_id": company_id,
        "company_name": company_name,
        "fact_type": str(raw.get("fact_type", "other"))[:100],
        "fact_value_text": str(raw.get("fact_value_text", ""))[:500] or None,
        "fact_value_numeric": numeric,
        "fact_currency": str(raw.get("fact_currency", ""))[:10] or None,
        "fact_unit": str(raw.get("fact_unit", ""))[:50] or None,
        "fact_year": year,
        "fact_quarter": str(raw.get("fact_quarter", ""))[:10] or None,
        "location_city": str(raw.get("location_city", ""))[:200] or None,
        "location_county": str(raw.get("location_county", ""))[:200] or None,
        "location_state": "Georgia",
        "oem_partner": str(raw.get("oem_partner", ""))[:500] or None,
        "confidence_score": confidence,
        "source_sentence": str(raw.get("source_sentence", ""))[:1000] or None,
        "extracted_at": datetime.utcnow(),
    }


def extract_facts(
    text: str,
    company_name: str,
    company_id: int | None = None,
    document_id: int | None = None,
) -> list[dict[str, Any]]:
    """
    Extract structured facts from document text using Ollama.

    Args:
        text: Document text (will be truncated to 3000 chars for LLM)
        company_name: Company context for the extraction prompt
        company_id: DB ID for linking
        document_id: DB ID for linking

    Returns:
        List of coerced fact dicts ready for DB insertion
    """
    cfg = Config.get()

    # Truncate text — 3000 chars is enough to capture key facts
    # while keeping LLM cost low
    truncated = text[:3000]

    prompt = _EXTRACTION_PROMPT.format(
        company_name=company_name,
        text=truncated,
    )

    try:
        raw_output = _call_ollama_sync(
            prompt=prompt,
            model=cfg.ollama_llm_model,
            base_url=cfg.ollama_base_url,
        )
    except httpx.HTTPStatusError as exc:
        logger.warning("Ollama entity extraction failed [%d] for %s", exc.response.status_code, company_name)
        return []
    except httpx.TimeoutException:
        logger.warning("Ollama entity extraction timed out for %s", company_name)
        return []
    except Exception as exc:
        logger.warning("Entity extraction error for %s: %s", company_name, exc)
        return []

    raw_facts = _parse_facts_json(raw_output)
    if not raw_facts:
        logger.debug("No facts extracted for %s", company_name)
        return []

    coerced = [_coerce_fact(f, company_id, company_name, document_id) for f in raw_facts]
    logger.info("Extracted %d facts for %s", len(coerced), company_name)
    return coerced


def save_facts_to_db(facts: list[dict[str, Any]]) -> int:
    """
    Insert extracted facts into gev_extracted_facts.
    Skips facts with very low confidence (< 0.3).
    Returns count of rows inserted.
    """
    if not facts:
        return 0

    session = get_session()
    inserted = 0
    try:
        for fact_data in facts:
            # Skip low-confidence extractions
            if (fact_data.get("confidence_score") or 0) < 0.3:
                continue
            # Skip facts with no numeric value AND no text value
            if not fact_data.get("fact_value_numeric") and not fact_data.get("fact_value_text"):
                continue
            fact = ExtractedFact(**fact_data)
            session.add(fact)
            inserted += 1
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return inserted
