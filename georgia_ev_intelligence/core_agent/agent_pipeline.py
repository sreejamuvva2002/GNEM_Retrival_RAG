"""
Phase 4 — Agent Pipeline (V4 Architecture).

Flow per question:

  1.  EXTRACT       — deterministic entity_extractor.extract()
  2.  CLASSIFY      — query_classifier.classify() → QueryClass
  3.  RESOLVE       — synonym_expander.resolve() against KB schema +
                      gev_domain_mapping_rules (approved rules only —
                      no heuristic invention)
  4.  BRANCH        — ambiguity_resolver.branches() emits 1 or 2
                      RetrievalBranch objects (KB-supported only)
  5.  RETRIEVE      — per branch, in parallel: SQL (sql_retriever),
                      Cypher (cypher_retriever), Qdrant dense + sparse
                      (qdrant_search). Each retriever is gated by
                      query_class.
  6.  MERGE         — retrieval_fusion.merge() dedupes by canonical id.
  7.  VALIDATE      — evidence_validator.validate_all() rejects rows that
                      disagree with the canonical gev_companies row or
                      that have no canonical row at all (unless the
                      query_class is FALLBACK_SEMANTIC or PRODUCT_CAPABILITY).
                      Validated rows are tagged `row['validated']=True`;
                      formatters assert this on entry.
  8.  RERANK        — only for PRODUCT_CAPABILITY / AMBIGUOUS / FALLBACK.
                      Operates on validated candidates only — cannot
                      resurrect a hard-filter rejection.
  9.  FUSE          — retrieval_fusion.fuse() applies the FINAL_SCORE_WEIGHTS
                      formula (shared/metadata_schema.py) with override
                      rules (SQL is sole source of truth for AGGREGATE /
                      COUNT / RANK / TOP_N / RISK).
  10. SELECT        — retrieval_fusion.select() per-class evidence policy.
  11. FORMAT/SYNTH  — deterministic formatter when branched / aggregate /
                      risk / top_n; LLM synthesis otherwise. Both paths
                      receive only validated evidence rows.
  12. VERIFY        — evidence_validator.verify_answer() flags hallucination
                      risk; downgrades support_level on failure.
  13. AUDIT         — audit_logger.write_audit() persists provenance to
                      gev_retrieval_audit + gev_retrieval_candidates.

The legacy V3 entry point (vector_retriever.retrieve_context) is no longer
called from this file. It remains importable as a fallback for tools that
have not been migrated yet.
"""
from __future__ import annotations

import argparse
import dataclasses
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from filters_and_validation.query_entity_extractor import extract, Entities
from filters_and_validation.query_classifier import classify
from filters_and_validation.synonym_expander import resolve as resolve_synonyms
from filters_and_validation.ambiguity_resolver import branches as build_branches
from retrievals.sql_retriever import run_plan as run_sql_plan
from retrievals.cypher_retriever import (
    run_plan as run_cypher_plan,
    render_cypher_query,
)
from retrievals.qdrant_search import dense as qdrant_dense, sparse as qdrant_sparse
from retrievals.retrieval_fusion import (
    apply_reranker_if_needed,
    fuse,
    merge,
    select,
)
from filters_and_validation.evidence_validator import validate_all, verify_answer
from core_agent.formatters import (
    format_aggregate,
    format_branched_answer,
    format_county_top,
    format_company_list,
    format_facility,
    format_oem_network,
    format_top_n,
)
from core_agent.retrieval_types import (
    AuditRecord,
    Candidate,
    QueryClass,
    RetrievalBranch,
)
from db_storage.audit_logger import (
    serialise_candidate_compact,
    write_audit,
)
from core_agent.streaming import stream_answer_collected
from shared.config import Config
from shared.logger import get_logger

logger = get_logger("phase4.agent")

_PRODUCT_CONTEXT_CHARS = 180


# ── Context formatting helpers ───────────────────────────────────────────────

def _table_cell(value: Any, max_chars: int | None = None) -> str:
    text = str(value or "").replace("|", "/")
    text = " ".join(text.split())
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _format_evidence_table(rows: list[dict[str, Any]]) -> str:
    """Pipe-separated table the LLM reads. Mirrors the V3 layout."""
    if not rows:
        return "No matching companies found."
    header = (
        "Company | Tier | Role | City | County | Employment | Primary_OEMs | "
        "Industry | EV_Relevant | Facility | Classification | Affiliation | Products"
    )
    divider = "-" * len(header)
    lines = [f"Total: {len(rows)} companies\n", header, divider]
    for c in rows:
        emp = int(float(c.get("employment") or 0)) if c.get("employment") else ""
        lines.append(
            " | ".join([
                _table_cell(c.get("company_name"), 48),
                _table_cell(c.get("tier")),
                _table_cell(c.get("ev_supply_chain_role"), 48),
                _table_cell(c.get("location_city")),
                _table_cell(c.get("location_county")),
                str(emp),
                _table_cell(c.get("primary_oems"), 48),
                _table_cell(c.get("industry_group"), 48),
                _table_cell(c.get("ev_battery_relevant")),
                _table_cell(c.get("facility_type"), 40),
                _table_cell(c.get("classification_method"), 48),
                _table_cell(c.get("supplier_affiliation_type"), 48),
                _table_cell(c.get("products_services"), _PRODUCT_CONTEXT_CHARS),
            ])
        )
    return "\n".join(lines)


def _evidence_rows(branch: RetrievalBranch) -> list[dict[str, Any]]:
    return [c.row for c in branch.evidence if getattr(c, "row", None)]


def _deterministic_format(
    branches_: list[RetrievalBranch],
    qclass: QueryClass,
    entities: Entities,
) -> str | None:
    """
    Build a deterministic answer string when the query class allows it.
    Returns None when LLM synthesis is required.
    """
    if len(branches_) >= 2:
        return format_branched_answer(branches_)

    if not branches_:
        return None
    rows = _evidence_rows(branches_[0])
    if not rows:
        return None

    if qclass == QueryClass.AGGREGATE:
        return format_aggregate(rows, tier=entities.tier)
    if qclass == QueryClass.TOP_N:
        return format_top_n(rows, n=int(getattr(entities, "top_n_limit", 10)), tier=entities.tier)
    if qclass == QueryClass.RANK:
        return format_top_n(rows, n=10, tier=entities.tier)
    if qclass == QueryClass.NETWORK and (entities.oem or entities.oem_list):
        oem = entities.oem or entities.oem_list[0]
        return format_oem_network(rows, oem=oem)
    if qclass == QueryClass.EXACT_FILTER and entities.county and len(rows) >= 1:
        # County-top is a special exact-filter sub-shape: rows already
        # sorted by employment desc.
        if rows[0].get("location_county"):
            return format_county_top(rows, county=entities.county)
    if qclass == QueryClass.EXACT_FILTER and entities.facility_type:
        return format_facility(rows, facility_type=entities.facility_type)
    if qclass == QueryClass.RISK:
        return format_company_list(rows, context_label="(single-supplier roles)")
    if qclass == QueryClass.COUNT:
        return f"{len(rows)} matching companies in the Georgia EV supply chain database."
    return None


# ── V3 backward-compatibility helpers ────────────────────────────────────────
# The eval / test harness imports these from pipeline. They are not used in
# the V4 flow but are preserved so callers do not need a coordinated upgrade.

def _parse_company_context(context: str) -> list[dict]:
    """Parse pipe-separated company table rows from formatted context."""
    companies: list[dict] = []
    in_table = False
    for line in context.splitlines():
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


def _parse_aggregate_context(context: str) -> list[dict]:
    """Parse formatted aggregate context (lines of '<county>: <n> employees (<m> companies)') back into row dicts."""
    import re
    rows: list[dict] = []
    for line in context.splitlines():
        m = re.match(r"\s*(.+?):\s+([\d,]+)\s+employees\s+\((\d+)\s+companies?\)", line)
        if m:
            rows.append({
                "county":           m.group(1).strip(),
                "total_employment": int(m.group(2).replace(",", "")),
                "company_count":    int(m.group(3)),
            })
    return rows


# ── Per-branch retrieval ─────────────────────────────────────────────────────

def _retrieve_for_branch(
    branch: RetrievalBranch,
    entities: Entities,
    qclass: QueryClass,
    question: str,
) -> list[Candidate]:
    """Run all enabled retrievers for one branch and return raw candidates."""
    per_source: list[list[Candidate]] = []

    if branch.sql_plan is not None:
        try:
            per_source.append(run_sql_plan(branch.sql_plan, branch.filters, entities))
        except Exception as exc:
            logger.warning("SQL retrieval failed: %s", exc)

    if branch.cypher_plan is not None:
        try:
            per_source.append(run_cypher_plan(branch.cypher_plan))
        except Exception as exc:
            logger.warning("Cypher retrieval failed: %s", exc)

    if branch.qdrant_plan is not None:
        try:
            dense_q = branch.qdrant_plan.semantic_query or question
            per_source.append(qdrant_dense(dense_q, entities, branch.filters,
                                           k=branch.qdrant_plan.k))
        except Exception as exc:
            logger.warning("Qdrant dense retrieval failed: %s", exc)
        try:
            sparse_q = branch.qdrant_plan.keyword_query or question
            per_source.append(qdrant_sparse(sparse_q, entities, branch.filters,
                                            k=branch.qdrant_plan.k))
        except Exception as exc:
            logger.warning("Qdrant sparse retrieval failed: %s", exc)

    return merge(per_source)


def _process_branch(
    branch: RetrievalBranch,
    entities: Entities,
    qclass: QueryClass,
    question: str,
) -> RetrievalBranch:
    """End-to-end branch processing: retrieve → validate → rerank → fuse → select."""
    candidates = _retrieve_for_branch(branch, entities, qclass, question)
    candidates = validate_all(candidates, entities, qclass, branch.filters)
    candidates = apply_reranker_if_needed(question, candidates, qclass)
    candidates = fuse(candidates, qclass)
    branch.evidence = select(candidates, qclass)
    branch.support_level = (
        "strong" if branch.evidence and any("sql" in c.sources or "cypher" in c.sources
                                            for c in branch.evidence)
        else "partial" if branch.evidence
        else "none"
    )
    return branch


# ── Audit helpers ────────────────────────────────────────────────────────────

def _summarise_sql_plan(plan) -> str | None:
    if plan is None:
        return None
    return f"SQLPlan(mode={plan.mode}, filters={plan.filters}, limit={plan.limit})"


def _all_candidates_for_audit(branches_: list[RetrievalBranch]) -> list[Candidate]:
    out: list[Candidate] = []
    for b in branches_:
        # Tag each candidate with its branch id for the audit row.
        for c in b.evidence:
            setattr(c, "branch_id", b.branch_id)
            out.append(c)
    return out


# ── Main agent ───────────────────────────────────────────────────────────────

class EVAgent:
    """Georgia EV Supply Chain Intelligence Agent — V4 architecture."""

    def __init__(self, model_override: str | None = None) -> None:
        cfg = Config.get()
        self.llm_model = model_override or cfg.ollama_llm_model
        logger.info("EVAgent V4 initialized | model=%s", self.llm_model)

    # ── V3 backward-compat helpers exposed on the agent ────────────────────
    @staticmethod
    def _format_companies(companies: list[dict]) -> str:
        """V3-compat: pipe-separated company table. Used by the eval harness."""
        return _format_evidence_table(companies)

    # ── Step 3: Generate ──────────────────────────────────────────────────
    def _generate(self, question: str, context: str) -> str:
        try:
            return stream_answer_collected(question, context, model=self.llm_model)
        except Exception as exc:
            logger.error("Synthesis failed: %s", exc)
            return f"[LLM unavailable] Retrieved data: {context[:500]}"

    # ── Inspect: full retrieval, no synthesis ─────────────────────────────
    def inspect(self, question: str) -> dict[str, Any]:
        start = time.monotonic()
        run_id = str(uuid.uuid4())
        logger.info("Question: %s", question[:100])

        entities = extract(question)
        qclass = classify(question, entities)
        logger.info("Classified: %s", qclass.value)

        resolved_terms = resolve_synonyms(
            terms=getattr(entities, "residual_abstract_terms", []) or [],
            entities=entities,
            question=question,
        )
        branches_ = build_branches(entities, resolved_terms, qclass, question)
        logger.info("Branches: %d", len(branches_))

        # Run branches in parallel — IO-bound, threads are sufficient.
        with ThreadPoolExecutor(max_workers=max(1, len(branches_))) as pool:
            futures = [
                pool.submit(_process_branch, b, entities, qclass, question)
                for b in branches_
            ]
            for fut in futures:
                fut.result()

        deterministic = _deterministic_format(branches_, qclass, entities)
        if deterministic is not None:
            retrieved_context = f"__DIRECT_ANSWER__:{deterministic}"
            prompt_context = ""
        else:
            # Concatenate evidence rows from all branches into a single table
            # for the LLM. This is only used when no deterministic formatter
            # applies (PRODUCT_CAPABILITY / FALLBACK / single-branch
            # EXACT_FILTER without a special formatter).
            rows: list[dict[str, Any]] = []
            for b in branches_:
                rows.extend(_evidence_rows(b))
            if rows:
                retrieved_context = (
                    f"[Selected evidence ({len(rows)} rows)]:\n"
                    + _format_evidence_table(rows)
                )
            else:
                retrieved_context = "No matching companies found."
            prompt_context = retrieved_context

        elapsed = time.monotonic() - start
        retrieved_count = sum(len(b.evidence) for b in branches_)
        logger.info(
            "Retrieved in %.1fs | rows=%d | branches=%d | class=%s",
            elapsed, retrieved_count, len(branches_), qclass.value,
        )

        # Audit hand-off — full record built here so ask() only adds the
        # answer text + verification before persisting.
        entity_dict = dataclasses.asdict(entities)
        entity_dict["retrieval_source"] = "v4"
        entity_dict["query_class"] = qclass.value

        retrieval_methods: set[str] = set()
        for b in branches_:
            for c in b.evidence:
                retrieval_methods |= c.sources
        sql_query = next(
            (_summarise_sql_plan(b.sql_plan) for b in branches_ if b.sql_plan),
            None,
        )
        cypher_query = next(
            (render_cypher_query(b.cypher_plan) for b in branches_ if b.cypher_plan),
            None,
        )
        dense_query = next(
            (b.qdrant_plan.semantic_query for b in branches_ if b.qdrant_plan),
            None,
        )
        sparse_query = next(
            (b.qdrant_plan.keyword_query for b in branches_ if b.qdrant_plan),
            None,
        )

        return {
            "question":          question,
            "run_id":            run_id,
            "query_class":       qclass.value,
            "retrieved_context": retrieved_context,
            "prompt_context":    prompt_context,
            "entities":          entity_dict,
            "branches":          branches_,
            "resolved_terms":    [r.as_dict() for r in resolved_terms],
            "retrieval_methods_used": sorted(retrieval_methods),
            "sql_query":         sql_query,
            "cypher_query":      cypher_query,
            "qdrant_dense_query": dense_query,
            "qdrant_sparse_query": sparse_query,
            "retrieved_count":   retrieved_count,
            "elapsed_s":         round(elapsed, 1),
        }

    # ── Ask: inspect + synthesise + verify + audit ────────────────────────
    def ask(self, question: str) -> dict[str, Any]:
        start = time.monotonic()
        result = self.inspect(question)
        context = result["retrieved_context"]

        if context.startswith("__DIRECT_ANSWER__:"):
            answer = context[len("__DIRECT_ANSWER__:"):]
        else:
            answer = self._generate(question, context)

        branches_: list[RetrievalBranch] = result["branches"]
        all_evidence = [c for b in branches_ for c in b.evidence]
        verification = verify_answer(answer, all_evidence)
        support_level = (
            "strong" if verification.status == "ok" and all_evidence
            else "partial" if verification.status == "risky" or all_evidence
            else "none"
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Answered in %.1fs | rows=%d | risk=%d (%s)",
            elapsed, result["retrieved_count"],
            verification.hallucination_risk, verification.status,
        )

        # Persist audit row + per-candidate rows.
        audit_record = AuditRecord(
            run_id=result["run_id"],
            question=question,
            query_class=result["query_class"],
            extracted_entities=result["entities"],
            hard_filters={b.branch_id: b.filters for b in branches_},
            ambiguous_terms=[r for r in result["resolved_terms"] if r.get("status") == "ambiguous"],
            selected_interpretations=[
                {"branch_id": b.branch_id, "meaning": b.interpreted_meaning}
                for b in branches_
            ],
            synonym_mappings=result["resolved_terms"],
            retrieval_methods_used=result["retrieval_methods_used"],
            sql_query=result["sql_query"],
            cypher_query=result["cypher_query"],
            qdrant_dense_query=result["qdrant_dense_query"],
            qdrant_sparse_query=result["qdrant_sparse_query"],
            final_evidence=[serialise_candidate_compact(c) for c in all_evidence],
            answer_text=answer,
            support_level=support_level,
            hallucination_risk=verification.hallucination_risk,
            audit_comment=verification.status,
            elapsed_ms=int(elapsed * 1000),
        )
        # Persist every candidate that any branch saw, including rejections.
        all_candidates = _all_candidates_for_audit(branches_)
        audit_id = write_audit(audit_record, all_candidates)

        result["answer"] = answer
        result["elapsed_s"] = round(elapsed, 1)
        result["support_level"] = support_level
        result["verification"] = verification.as_dict()
        result["audit_id"] = audit_id
        # Drop non-serialisable objects so the result dict is JSON-safe.
        result.pop("branches", None)
        result["candidates_summary"] = {
            "selected": sum(1 for c in all_candidates if c.final_selected),
            "rejected": sum(1 for c in all_candidates if not c.hard_filter_passed),
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask one Georgia EV question from the terminal.")
    parser.add_argument("--question", required=True, help="Single question to run through the V4 agent.")
    parser.add_argument("--model", default=None, help="Optional Ollama model override for synthesis.")
    parser.add_argument(
        "--context-only", action="store_true",
        help="Print only the exact retrieved context block inserted into the synthesis prompt.",
    )
    args = parser.parse_args()

    agent = EVAgent(model_override=args.model)
    if args.context_only:
        result = agent.inspect(args.question)
        print(result["prompt_context"] or result["retrieved_context"], end="")
        return

    result = agent.ask(args.question)
    print(result["answer"])


if __name__ == "__main__":
    main()
