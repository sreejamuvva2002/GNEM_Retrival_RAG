"""
Phase 4 — Agent Pipeline (V3 Architecture)

  Step 1: EXTRACT  — Deterministic entity extraction (no LLM)
  Step 2: RETRIEVE — Text-to-SQL (LLM generates SQL) + Deterministic Cypher + Gemma fallback
  Step 3: GENERATE — Single LLM synthesis call (qwen2.5:7b)

WHY TEXT-TO-SQL REPLACES HARDCODED RULES:
  Rules break for edge cases:
    "Tier 1 only"       → needs exact match, not ilike
    "at least 500 emp"  → needs >= filter
    "top 5 by county"   → needs LIMIT 5
    "excluding OEMs"    → needs NOT LIKE filter

  The LLM reads the question + schema → generates correct SQL for ANY phrasing.
  No code changes needed when new question patterns appear.
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from typing import Any

from phase4_agent.entity_extractor import extract, Entities
from phase4_agent.streaming import (
    count_context_rows,
    prompt_context_for_model,
    stream_answer_collected,
)
from phase4_agent.vector_retriever import retrieve_context
# from phase4_agent.formatters import ...   ← available for future use, not active
from shared.config import Config
from shared.logger import get_logger

logger = get_logger("phase4.agent")

_MAX_LLM_COMPANIES = 15   # capped from 50 — prevents LLM reading overload that causes 'not found' hallucination
_PRODUCT_CONTEXT_CHARS = 180


def _table_cell(value: Any, max_chars: int | None = None) -> str:
    """Format one pipe-table cell without letting source pipes corrupt columns."""
    text = str(value or "").replace("|", "/")
    text = " ".join(text.split())
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _is_oem_reference(tier_value: str | None) -> bool:
    """True when 'OEM' was extracted as a bare acronym, not as a real tier label."""
    if not tier_value:
        return False
    if tier_value.lower() in ("oem supply chain", "oem footprint", "oem (footprint)"):
        return False
    return tier_value.strip().upper() == "OEM"


def _parse_aggregate_context(context: str) -> list[dict]:
    """
    Parse formatted aggregate context back into row dicts.
    Context lines look like: "Troup County: 2,280 employees (7 companies)"
    """
    import re
    rows = []
    for line in context.splitlines():
        m = re.match(r"\s*(.+?):\s+([\d,]+)\s+employees\s+\((\d+)\s+companies?\)", line)
        if m:
            rows.append({
                "county":           m.group(1).strip(),
                "total_employment": int(m.group(2).replace(",", "")),
                "company_count":    int(m.group(3)),
            })
    return rows


def _parse_company_context(context: str) -> list[dict]:
    """
    Parse pipe-separated company table rows from formatted context.
    Context rows look like:
    "Company | Tier | Role | City | County | Emp | OEMs | Industry | EV | Facility | Classification | Affiliation | Products"
    """
    companies = []
    lines = context.splitlines()
    in_table = False
    for line in lines:
        if line.startswith("Company | Tier"):
            in_table = True
            continue
        if in_table and line.startswith("---"):
            continue
        if in_table and "|" in line and not line.startswith("Total:"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                companies.append({
                    "company_name":         parts[0],
                    "tier":                 parts[1],
                    "ev_supply_chain_role": parts[2],
                    "location_city":        parts[3] if len(parts) > 3 else "",
                    "location_county":      parts[4] if len(parts) > 4 else "",
                    "employment":           parts[5] if len(parts) > 5 and parts[5] else None,
                    "primary_oems":         parts[6] if len(parts) > 6 else "",
                    "industry_group":       parts[7] if len(parts) > 7 else "",
                    "ev_battery_relevant":  parts[8] if len(parts) > 8 else "",
                    "facility_type":        parts[9] if len(parts) > 9 else "",
                    "classification_method": parts[10] if len(parts) > 10 else "",
                    "supplier_affiliation_type": parts[11] if len(parts) > 11 else "",
                    "products_services":    parts[12] if len(parts) > 12 else "",
                })
    return companies


def _sql_row_to_company(row: dict) -> dict:
    """
    Map a raw SQL result row to a company dict for _format_companies.

    WHY SCHEMA-AGNOSTIC (not a hardcoded column whitelist):
      If a new column is added to gev_companies (e.g. latitude, naics_code,
      supplier_tier_2_of, etc.) it is automatically passed through here.
      _format_companies reads specific named keys and ignores extras, so
      new columns are available in context without any code change needed.

      Rule: pass ALL row keys through, coercing None -> empty string.
    """
    return {k: (v if v is not None else "") for k, v in row.items()}


class EVAgent:
    """Georgia EV Supply Chain Intelligence Agent — V3 Architecture."""

    def __init__(self, model_override: str | None = None) -> None:
        cfg = Config.get()
        # model_override lets the evaluator pass 'gemma2:9b', 'qwen2.5:14b' etc.
        # directly without modifying env vars (Config is @lru_cache — env changes are ignored).
        self.llm_model = model_override or cfg.ollama_llm_model
        logger.info("EVAgent initialized | model=%s", self.llm_model)


    # ── Step 3: Generate ──────────────────────────────────────────────────────

    def _generate(self, question: str, context: str) -> str:
        """Synthesize answer — uses self.llm_model (supports model_override for eval)."""
        try:
            return stream_answer_collected(question, context, model=self.llm_model)
        except Exception as exc:
            logger.error("Synthesis failed: %s", exc)
            return f"[LLM unavailable] Retrieved data: {context[:500]}"


    # ── Step 2: Retrieve ──────────────────────────────────────────────────────

    def _retrieve(self, question: str, e: Entities) -> tuple[str, bool]:
        results = retrieve_context(question, e)
        if isinstance(results, str):
            return results, False
        if results:
            return f"[Qdrant — Company records ({len(results)} results)]:\n" + self._format_companies(results), False
        return "No matching companies found.", False

    # ── Formatting helper ─────────────────────────────────────────────────────

    @staticmethod
    def _format_companies(companies: list[dict]) -> str:
        """Compact pipe-separated table. ~80 chars/row, LLM-readable."""
        if not companies:
            return "No matching companies found."

        header  = (
            "Company | Tier | Role | City | County | Employment | Primary_OEMs | "
            "Industry | EV_Relevant | Facility | Classification | Affiliation | Products"
        )
        divider = "-" * len(header)
        rows    = [f"Total: {len(companies)} companies\n", header, divider]

        for c in companies:
            name     = _table_cell(c.get("company_name"), 48)
            tier     = _table_cell(c.get("tier"))
            role     = _table_cell(c.get("ev_supply_chain_role"), 48)
            city     = _table_cell(c.get("location_city"))
            county   = _table_cell(c.get("location_county"))
            emp      = int(float(c.get("employment") or 0)) if c.get("employment") else ""
            oems     = _table_cell(c.get("primary_oems"), 48)
            industry = _table_cell(c.get("industry_group"), 48)
            ev_rel   = _table_cell(c.get("ev_battery_relevant"))
            facility = _table_cell(c.get("facility_type"), 40)
            classification = _table_cell(c.get("classification_method"), 48)
            affiliation = _table_cell(c.get("supplier_affiliation_type"), 48)
            product  = _table_cell(c.get("products_services"), _PRODUCT_CONTEXT_CHARS)
            rows.append(
                f"{name} | {tier} | {role} | {city} | {county} | {emp} | {oems} | "
                f"{industry} | {ev_rel} | {facility} | {classification} | "
                f"{affiliation} | {product}"
            )
        return "\n".join(rows)

    # ── Main entry point ──────────────────────────────────────────────────────

    def inspect(self, question: str) -> dict[str, Any]:
        """Retrieve data for one question without running final answer synthesis."""
        start = time.monotonic()
        logger.info("Question: %s", question[:100])

        # Step 1: Extract entities
        entities = extract(question)
        logger.info(
            "Extracted: tier=%s county=%s oem=%s industry_group=%s role=%s | "
            "aggregate=%s risk=%s oem_dep=%s capacity=%s misalign=%s top_n=%s",
            entities.tier, entities.county, entities.oem, entities.industry_group,
            entities.ev_role or entities.ev_role_list,
            entities.is_aggregate, entities.is_risk_query, entities.is_oem_dependency,
            entities.is_capacity_risk, entities.is_misalignment, entities.is_top_n,
        )

        # Step 2: Retrieve (Qdrant-only)
        context, cypher_used = self._retrieve(question, entities)
        retrieved_count = count_context_rows(prompt_context_for_model(context))
        prompt_context = ""
        if not context.startswith("__DIRECT_ANSWER__:"):
            prompt_context = prompt_context_for_model(context)

        elapsed = time.monotonic() - start
        logger.info("Retrieved in %.1fs | rows=%d | cypher=%s", elapsed, retrieved_count, cypher_used)

        # Serialize ALL entity fields automatically using dataclasses.asdict().
        # WHY: a hardcoded dict silently drops new fields added to Entities.
        # dataclasses.asdict() always includes every field — zero maintenance.
        entity_dict = dataclasses.asdict(entities)
        entity_dict["cypher_used"] = cypher_used   # pipeline-level flag, not in dataclass
        entity_dict["retrieval_source"] = "vector"

        return {
            "question":          question,
            "retrieved_context": context,
            "prompt_context":    prompt_context,
            "entities":          entity_dict,
            "retrieved_count":   retrieved_count,
            "elapsed_s":         round(elapsed, 1),
        }

    def ask(self, question: str) -> dict[str, Any]:
        """
        Answer a question about the Georgia EV supply chain.

        Architecture:
          Retrieval : 0-1 LLM calls (deterministic SQL/Cypher or Text-to-SQL generation)
          Synthesis : 1 LLM call always (true RAG — LLM reads data and writes answer)

        With streaming enabled, complex 2-call queries (~25s total) feel instant
        because the first token arrives in <2s and the user reads as it generates.

        LLM call count:
          Risk query             → 0 retrieval + 1 synthesis = 1 call  (~10s)
          Deterministic SQL/Neo4j→ 0 retrieval + 1 synthesis = 1 call  (~10s)
          Text-to-SQL (complex)  → 1 SQL gen   + 1 synthesis = 2 calls (~25s, streamed)
        """
        start = time.monotonic()
        result = self.inspect(question)
        context = result["retrieved_context"]
        cypher_used = result["entities"].get("cypher_used", False)

        # Step 3: Synthesize — LLM reads data and writes the answer (RAG)
        # Exception: SPOF risk query is already a formatted answer (direct list)
        if context.startswith("__DIRECT_ANSWER__:"):
            answer = context[len("__DIRECT_ANSWER__:"):]
        else:
            answer = self._generate(question, context)

        elapsed = time.monotonic() - start
        logger.info("Answered in %.1fs | rows=%d | cypher=%s", elapsed, result["retrieved_count"], cypher_used)

        result["answer"] = answer
        result["elapsed_s"] = round(elapsed, 1)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask one Georgia EV question from the terminal.")
    parser.add_argument(
        "--question",
        required=True,
        help="Single question to run through the Phase 4 agent.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Ollama model override for answer synthesis.",
    )
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Print only the exact retrieved context block inserted into the synthesis prompt.",
    )
    args = parser.parse_args()

    agent = EVAgent(model_override=args.model)
    if args.context_only:
        result = agent.inspect(args.question)
        print(result["prompt_context"], end="")
        return

    result = agent.ask(args.question)
    print(result["answer"])


if __name__ == "__main__":
    main()
