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

import time
from typing import Any

from phase4_agent.entity_extractor import extract, Entities
from phase4_agent.sql_retriever import (
    query_companies,
    full_text_search,
    get_single_supplier_roles,
    aggregate_employment_by_county,
    top_companies_by_employment,
)
from phase4_agent.text_to_sql import text_to_sql
from phase4_agent.text_to_cypher import (
    execute_cypher_safe,
    execute_cypher,
    normalize_cypher_results,
)
from phase4_agent.cypher_builder import build_cypher
from phase4_agent.streaming import stream_answer_collected
# from phase4_agent.formatters import ...   ← available for future use, not active
from shared.config import Config
from shared.logger import get_logger

logger = get_logger("phase4.agent")

_MAX_LLM_COMPANIES = 15   # capped from 50 — prevents LLM reading overload that causes 'not found' hallucination


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
    Context rows look like: "Company | Tier | Role | County | Emp | Industry | EV | Facility | Products"
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
                    "location_county":      parts[3],
                    "employment":           parts[4] if parts[4] else None,
                    "industry_group":       parts[5] if len(parts) > 5 else "",
                    "ev_battery_relevant":  parts[6] if len(parts) > 6 else "",
                    "facility_type":        parts[7] if len(parts) > 7 else "",
                    "products_services":    parts[8] if len(parts) > 8 else "",
                })
    return companies


def _sql_row_to_company(row: dict) -> dict:
    """Map a raw SQL result row to the standard company dict for _format_companies."""
    return {
        "company_name":         row.get("company_name", ""),
        "tier":                 row.get("tier", ""),
        "ev_supply_chain_role": row.get("ev_supply_chain_role", ""),
        "location_county":      row.get("location_county", ""),
        "employment":           row.get("employment", ""),
        "industry_group":       row.get("industry_group", ""),
        "ev_battery_relevant":  row.get("ev_battery_relevant", ""),
        "facility_type":        row.get("facility_type", ""),
        "products_services":    row.get("products_services", ""),
        "primary_oems":         row.get("primary_oems", ""),
    }


class EVAgent:
    """Georgia EV Supply Chain Intelligence Agent — V3 Architecture."""

    def __init__(self) -> None:
        cfg = Config.get()
        self.llm_model = cfg.ollama_llm_model
        logger.info("EVAgent initialized | model=%s", self.llm_model)

    # ── Step 3: Generate ──────────────────────────────────────────────────────

    def _generate(self, question: str, context: str) -> str:
        """
        Synthesize a natural language answer from retrieved context.
        Uses streaming internally (stream=True) — better per-token timeout handling.
        Collects tokens into a string for backward compatibility.
        """
        try:
            return stream_answer_collected(question, context)
        except Exception as exc:
            logger.error("Synthesis failed: %s", exc)
            return f"[LLM unavailable] Retrieved data: {context[:500]}"

    # ── Step 2: Retrieve ──────────────────────────────────────────────────────

    def _retrieve(self, question: str, e: Entities) -> tuple[str, bool]:
        """
        V3 Architecture — priority order:

        A. Risk query      → Direct DB call, no LLM (always a list of roles)
        B. Text-to-SQL     → LLM generates full SQL for aggregate/county/OEM/filter Qs
                             Handles 'only', 'at least', 'top N', 'excluding' automatically
        C. Det. Cypher     → Entity-based Cypher (role, facility, product, company)
        D. Gemma Cypher    → Free-form graph questions (last LLM resort)
        E. Full-text       → PostgreSQL keyword fallback

        Returns (context_string, cypher_was_used)
        """

        # ── A: Risk query subtypes — deterministic, no LLM retrieval ─────────────
        #
        # A1: SPOF (single-point-of-failure) — Q27 pattern
        #     "roles served by only one company" → return list directly
        if e.is_risk_query:
            rows = get_single_supplier_roles()
            if not rows:
                return (
                    "No single-supplier roles found — every EV supply chain role "
                    "in Georgia has at least 2 companies covering it."
                ), False
            lines = [
                f"• {r['company']} [{r['tier']}] — sole supplier for: {r['role']}"
                for r in rows
            ]
            answer = (
                f"Georgia has {len(rows)} EV supply chain roles served by only ONE company "
                f"(single points of failure):\n\n" + "\n".join(lines)
            )
            return f"__DIRECT_ANSWER__:{answer}", False

        # A2: OEM dependency — Q28 pattern
        #     "Battery Cell/Pack suppliers sole-sourced by specific OEM"
        if e.is_oem_dependency:
            role_filter = "Battery Cell OR Battery Pack"
            if e.ev_role:
                role_filter = e.ev_role
            elif e.ev_role_list:
                role_filter = " OR ".join(e.ev_role_list)
            filters: dict = {"ev_supply_chain_role": role_filter}
            # CRITICAL: Do NOT apply tier filter here if e.tier was triggered by the
            # phrase "OEM" appearing in the question ("single OEM", "by an OEM") —
            # those are OEM-reference words, not meaning the companies are OEM-tier.
            # Battery Cell/Pack companies are always Tier 2/3, not tier='OEM'.
            if e.tier and not _is_oem_reference(e.tier):
                filters["tier"] = e.tier
            pg_rows = query_companies(filters=filters, limit=_MAX_LLM_COMPANIES)
            if pg_rows:
                ctx = f"[SQL — OEM Dependency Risk ({len(pg_rows)} suppliers)]:\n" + self._format_companies(pg_rows)
                return ctx, False

        # A3: Capacity risk — Q29/Q32 pattern
        #     "suppliers with fewer than N employees"
        if e.is_capacity_risk:
            filters = {}
            if e.oem:
                filters["primary_oems"] = e.oem
            if e.tier:
                filters["tier"] = e.tier
            if e.ev_role:
                filters["ev_supply_chain_role"] = e.ev_role
            elif e.ev_role_list:
                filters["ev_supply_chain_role"] = " OR ".join(e.ev_role_list)
            if e.max_employment:
                filters["max_employment"] = e.max_employment
            pg_rows = query_companies(filters=filters, limit=_MAX_LLM_COMPANIES)
            if pg_rows:
                ctx = f"[SQL — Capacity Risk ({len(pg_rows)} small suppliers)]:\n" + self._format_companies(pg_rows)
                return ctx, False

        # A4: Supply chain misalignment — Q30 pattern
        #     "Tier 2/3 + EV Relevant + General Automotive + Battery supply"
        if e.is_misalignment:
            filters = {}
            if e.tier:
                filters["tier"] = e.tier
            if e.ev_role:
                filters["ev_supply_chain_role"] = e.ev_role
            # EV relevant = Yes
            filters["ev_battery_relevant"] = "Yes"
            pg_rows = query_companies(filters=filters, limit=_MAX_LLM_COMPANIES)
            if pg_rows:
                ctx = f"[SQL — Misalignment Risk ({len(pg_rows)} suppliers)]:\n" + self._format_companies(pg_rows)
                return ctx, False

        # ── B: Deterministic SQL fast-paths for clear extracted entities ──────────

        if e.is_top_n:
            tier_val = e.tier if e.tier and not _is_oem_reference(e.tier) else None
            rows = top_companies_by_employment(limit=e.top_n_limit, tier=tier_val)
            if rows:
                ctx = f"[SQL — Top {e.top_n_limit} companies by employment]:\n" + self._format_companies(rows)
                return ctx, False

        if e.industry_group:
            # "Chemicals and Allied Products" → direct SQL filter — no LLM needed
            filters: dict = {"industry_group": e.industry_group}
            if e.tier:
                filters["tier"] = e.tier
            if e.ev_role:
                filters["ev_supply_chain_role"] = e.ev_role
            pg_rows = query_companies(filters=filters, limit=_MAX_LLM_COMPANIES)
            if pg_rows:
                ctx = f"[SQL — Industry group '{e.industry_group}' ({len(pg_rows)} found)]:\n" + self._format_companies(pg_rows)
                return ctx, False

        if e.county and not e.is_aggregate:
            pg_rows = query_companies(filters={"location_county": e.county}, limit=_MAX_LLM_COMPANIES)
            if pg_rows:
                pg_rows.sort(key=lambda x: float(x.get("employment") or 0), reverse=True)
                ctx = f"[SQL — Companies in {e.county} ({len(pg_rows)} found)]:\n" + self._format_companies(pg_rows)
                return ctx, False

        if e.oem and not _is_oem_reference(e.oem) and not e.is_aggregate:
            # Compound filter: OEM + optional employment range + optional tier
            oem_filters: dict = {"primary_oems": e.oem}
            if e.tier and not _is_oem_reference(e.tier):
                oem_filters["tier"] = e.tier
            if e.min_employment:
                oem_filters["min_employment"] = e.min_employment
            if e.max_employment:
                oem_filters["max_employment"] = e.max_employment
            pg_rows = query_companies(filters=oem_filters, limit=_MAX_LLM_COMPANIES)
            if pg_rows:
                ctx = f"[SQL — Suppliers to {e.oem} ({len(pg_rows)} found)]:\n" + self._format_companies(pg_rows)
                return ctx, False

        if e.is_aggregate and e.tier and not _is_oem_reference(e.tier):
            data = aggregate_employment_by_county(tier=e.tier)
            if data:
                lines = [
                    f"{r['county']}: {int(r['total_employment']):,} employees ({r['company_count']} companies)"
                    for r in data[:30]
                ]
                ctx = f"[SQL — Employment by county ({e.tier})]:\n" + "\n".join(lines)
                return ctx, False

        # Compound filter: tier + role (or role list) + optional employment range
        if (e.tier or e.ev_role or e.ev_role_list or e.min_employment or e.max_employment
                or e.facility_type) and not e.is_aggregate:
            filters = {}
            if e.tier:
                filters["tier"] = e.tier
            if e.ev_role:
                filters["ev_supply_chain_role"] = e.ev_role
            elif e.ev_role_list:
                filters["ev_supply_chain_role"] = " OR ".join(e.ev_role_list)
            if e.min_employment:
                filters["min_employment"] = e.min_employment
            if e.max_employment:
                filters["max_employment"] = e.max_employment
            if e.facility_type:
                filters["facility_type"] = e.facility_type
            if filters:
                pg_rows = query_companies(filters=filters, limit=_MAX_LLM_COMPANIES)
                if pg_rows:
                    ctx = f"[SQL — Filtered ({len(pg_rows)} results)]:\n" + self._format_companies(pg_rows)
                    return ctx, False

        # ── C: Text-to-SQL (for complex / ambiguous queries only) ────────────────
        # Called when: aggregate with no clear tier, multi-constraint questions,
        # range filters (>500 emp), or phrases like 'excluding', 'at least'.
        # Adds ~13s overhead — only justified when deterministic SQL can't handle it.
        if e.is_aggregate or e.is_top_n or e.county or (e.oem and not _is_oem_reference(e.oem)):
            logger.info("Using Text-to-SQL for complex query: %s", question[:80])
            sql_rows = text_to_sql(question)
            if sql_rows:
                first = sql_rows[0]
                if "total_employment" in first:
                    lines = [
                        f"{r.get('location_county', r.get('county', ''))}: "
                        f"{int(r.get('total_employment', 0)):,} employees "
                        f"({r.get('company_count', '')} companies)"
                        for r in sql_rows[:30]
                    ]
                    ctx = f"[Text-to-SQL — {len(sql_rows)} counties]:\n" + "\n".join(lines)
                else:
                    companies = [_sql_row_to_company(r) for r in sql_rows]
                    ctx = f"[Text-to-SQL — {len(companies)} results]:\n" + self._format_companies(companies)
                return ctx, False

        # ── C: Deterministic Cypher from extracted entities ────────────────────
        built_cypher = build_cypher(e, question)
        if built_cypher:
            try:
                raw_rows = execute_cypher(built_cypher)
                # Product keyword Cypher uses AND (precise). If 0 results,
                # fall back to OR using the fallback clause embedded in the query.
                if not raw_rows and "// FALLBACK_OR:" in built_cypher:
                    fallback_clause = built_cypher.split("// FALLBACK_OR:")[-1].strip()
                    fallback_cypher = (
                        "MATCH (c:Company)\n"
                        f"WHERE {fallback_clause}\n"
                        "RETURN c.name AS company_name, c.tier, c.ev_supply_chain_role,\n"
                        "       c.employment, c.location_county, c.facility_type,\n"
                        "       c.ev_battery_relevant, c.industry_group, c.products_services\n"
                        "ORDER BY c.employment DESC LIMIT 50"
                    )
                    logger.info("AND Cypher returned 0 — retrying with OR fallback")
                    raw_rows = execute_cypher(fallback_cypher)

                if raw_rows:
                    companies = normalize_cypher_results(raw_rows)
                    ctx = (
                        f"[Neo4j — Deterministic Cypher ({len(companies)} results)]:\n"
                        + self._format_companies(companies)
                    )
                    return ctx, True
            except Exception as exc:
                logger.warning("Deterministic Cypher failed: %s — trying Gemma", exc)

        # ── D: Gemma Text-to-Cypher (last LLM resort) ─────────────────────────
        # For free-form graph questions that no entity pattern covers.
        logger.info("Using Text-to-Cypher (Gemma) for: %s", question[:80])
        cypher_rows = execute_cypher_safe(question)
        if cypher_rows:
            companies = normalize_cypher_results(cypher_rows)
            ctx = f"[Neo4j — Gemma Cypher ({len(companies)} results)]:\n" + self._format_companies(companies)
            return ctx, True

        # ── E: PostgreSQL full-text fallback ───────────────────────────────────
        logger.info("All paths returned 0 — falling back to full-text search")
        fallback_tier = e.tier if e.tier and not _is_oem_reference(e.tier) else None
        pg_rows = full_text_search(question.split(), tier=fallback_tier, limit=_MAX_LLM_COMPANIES)
        if pg_rows:
            ctx = f"[PostgreSQL full-text — {len(pg_rows)} results]:\n" + self._format_companies(pg_rows)
            return ctx, False

        # ── F: Nothing found ───────────────────────────────────────────────────
        return (
            "No relevant data found. The database contains 193 Georgia EV supply chain "
            "companies with tier, role, employment, county, OEM, industry, and facility type data."
        ), False

    # ── Formatting helper ─────────────────────────────────────────────────────

    @staticmethod
    def _format_companies(companies: list[dict]) -> str:
        """Compact pipe-separated table. ~80 chars/row, LLM-readable."""
        if not companies:
            return "No matching companies found."

        header  = "Company | Tier | Role | County | Employment | Industry | EV_Relevant | Facility | Products"
        divider = "-" * len(header)
        rows    = [f"Total: {len(companies)} companies\n", header, divider]

        for c in companies:
            name     = (c.get("company_name") or "")[:40]
            tier     = (c.get("tier") or "")
            role     = (c.get("ev_supply_chain_role") or "")[:35]
            county   = (c.get("location_county") or "")
            emp      = int(float(c.get("employment") or 0)) if c.get("employment") else ""
            industry = (c.get("industry_group") or "")[:30]
            ev_rel   = (c.get("ev_battery_relevant") or "")
            facility = (c.get("facility_type") or "")
            product  = (c.get("products_services") or "")[:60]
            rows.append(
                f"{name} | {tier} | {role} | {county} | {emp} | "
                f"{industry} | {ev_rel} | {facility} | {product}"
            )
        return "\n".join(rows)

    # ── Main entry point ──────────────────────────────────────────────────────

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

        # Step 2: Retrieve
        context, cypher_used = self._retrieve(question, entities)
        # Count pipe-separated company rows (more accurate than line count)
        retrieved_count = sum(1 for line in context.splitlines() if " | " in line and not line.startswith("Company"))

        # Step 3: Synthesize — LLM reads data and writes the answer (RAG)
        # Exception: SPOF risk query is already a formatted answer (direct list)
        if context.startswith("__DIRECT_ANSWER__:"):
            answer = context[len("__DIRECT_ANSWER__:"):]
        else:
            answer = self._generate(question, context)

        elapsed = time.monotonic() - start
        logger.info("Answered in %.1fs | rows=%d | cypher=%s", elapsed, retrieved_count, cypher_used)

        return {
            "question":          question,
            "answer":            answer,
            "retrieved_context": context,   # ← for RAGAS faithfulness/precision/recall
            "entities": {
                "tier":           entities.tier,
                "county":         entities.county,
                "company":        entities.company_name,
                "oem":            entities.oem,
                "industry_group": entities.industry_group,
                "role":           entities.ev_role or entities.ev_role_list,
                "keywords":       entities.product_keywords,
                "aggregate":      entities.is_aggregate,
                "risk_query":     entities.is_risk_query,
                "oem_dependency": entities.is_oem_dependency,
                "capacity_risk":  entities.is_capacity_risk,
                "misalignment":   entities.is_misalignment,
                "cypher_used":    cypher_used,
            },
            "retrieved_count":   retrieved_count,
            "elapsed_s":         round(elapsed, 1),
        }

