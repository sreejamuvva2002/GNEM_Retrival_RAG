"""
Phase 1 — Main Pipeline Orchestrator
End-to-end: GNEM Excel → Search → Extract → Store → Structured Facts

Architecture:
  1. Load 205 companies from GNEM Excel → sync to PostgreSQL
  2. For each company (async, 5 companies at a time):
     a. Generate Tavily search queries
     b. Run Tavily search → get URLs
     c. For each URL: extract text (Tavily Extract or PyMuPDF)
     d. Save to B2 + PostgreSQL
     e. Run entity extraction → structured facts → PostgreSQL

Usage:
  # Full run (all 205 companies):
  python -m phase1_extraction.pipeline

  # Test run (first N companies):
  python -m phase1_extraction.pipeline --limit 3

  # Single company (by name):
  python -m phase1_extraction.pipeline --company "Hanwha Q Cells"

  # Skip extraction, only load GNEM to DB:
  python -m phase1_extraction.pipeline --load-only
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from phase1_extraction.doc_storage import (
    get_document_count_for_company,
    mark_document_failed,
    save_document,
)
from phase1_extraction.entity_extractor import extract_facts, save_facts_to_db
from phase1_extraction.extractor import extract_document
from phase1_extraction.kb_loader import (
    get_all_companies_from_db,
    load_companies_from_excel,
    sync_companies_to_db,
)
from phase1_extraction.query_generator import build_queries, estimate_query_count
from phase1_extraction.searcher import search_company
from shared.logger import get_logger

logger = get_logger("phase1.pipeline")

# How many companies to process in parallel
# 3 = safe concurrency for Tavily on one machine (no Ollama blocking now)
COMPANY_CONCURRENCY = 3

# Max URLs to extract per company per run
# 8 = quality over quantity (research-backed: 500 quality docs > 50k noisy ones)
MAX_URLS_PER_COMPANY = 8

# Minimum Tavily relevance score to bother extracting
# Filters out directory listings, social media, irrelevant results
MIN_RELEVANCE_SCORE = 0.4


async def process_company(
    company: dict[str, Any],
    skip_if_has_docs: bool = True,
) -> dict[str, Any]:
    """
    Full pipeline for one company.

    Returns a result summary dict.
    """
    company_name = company.get("company_name", "")
    company_id = company.get("id")
    start = time.monotonic()

    result = {
        "company": company_name,
        "urls_found": 0,
        "docs_extracted": 0,
        "docs_failed": 0,
        "facts_extracted": 0,
        "skipped": False,
        "error": None,
    }

    # Skip if already has enough documents (idempotent re-runs)
    if skip_if_has_docs:
        existing_count = get_document_count_for_company(company_name)
        if existing_count >= 5:
            logger.info("Skipping '%s' — already has %d documents", company_name, existing_count)
            result["skipped"] = True
            return result

    try:
        # Step 1: Generate queries
        queries = build_queries(company)
        logger.info("[%s] Generated %d queries", company_name, len(queries))

        # Step 2: Search with Tavily
        url_results = await search_company(company, queries)
        result["urls_found"] = len(url_results)

        # Filter by relevance score — skip low-quality results (directories, social media)
        url_results = [
            u for u in url_results
            if u.get("score", 0.0) >= MIN_RELEVANCE_SCORE
        ]
        # Take top N by relevance score
        url_results = sorted(url_results, key=lambda u: u.get("score", 0), reverse=True)
        url_results = url_results[:MAX_URLS_PER_COMPANY]
        logger.info(
            "[%s] %d URLs after relevance filter (score >= %.1f)",
            company_name, len(url_results), MIN_RELEVANCE_SCORE
        )

        # Step 3: Extract content from each URL
        for url_result in url_results:
            url = url_result["url"]

            extracted = await extract_document(
                url=url,
                company_name=company_name,
            )

            if extracted.error or not extracted.text:
                mark_document_failed(url, extracted.error, company_id, company_name)
                result["docs_failed"] += 1
                continue

            # Step 4: Save raw document to B2 + PostgreSQL
            # NOTE: Entity extraction (facts) is intentionally REMOVED from Phase 1.
            # Root cause: Ollama queues all requests — 5 concurrent companies × 20 docs
            # = 100 queued Ollama calls → each waits 99× inference time → timeouts.
            # Resolution: Phase 3 (Neo4j) extracts structured facts from the top 3 docs
            # per company using sequential (not concurrent) Ollama calls.
            doc_id = save_document(
                extracted=extracted,
                company_id=company_id,
                search_query=url_result.get("snippet", "")[:500],
                relevance_score=url_result.get("score", 0.0),
            )

            if doc_id is None:
                result["docs_failed"] += 1
                continue

            result["docs_extracted"] += 1

    except Exception as exc:
        logger.error("Pipeline error for '%s': %s", company_name, exc, exc_info=True)
        result["error"] = str(exc)[:200]

    elapsed = time.monotonic() - start
    logger.info(
        "[%s] Done in %.1fs — %d URLs | %d docs | %d facts",
        company_name, elapsed,
        result["urls_found"], result["docs_extracted"], result["facts_extracted"],
    )
    return result


async def run_pipeline(
    companies: list[dict[str, Any]],
    concurrency: int = COMPANY_CONCURRENCY,
    skip_if_has_docs: bool = True,
) -> list[dict[str, Any]]:
    """
    Process multiple companies with async concurrency.
    Returns list of result summaries.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    async def _process(company: dict[str, Any]) -> None:
        async with semaphore:
            res = await process_company(company, skip_if_has_docs=skip_if_has_docs)
            results.append(res)

    tasks = [_process(c) for c in companies]
    await asyncio.gather(*tasks)
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a clear summary of the pipeline run."""
    total = len(results)
    skipped = sum(1 for r in results if r["skipped"])
    processed = total - skipped
    total_docs = sum(r["docs_extracted"] for r in results)
    total_facts = sum(r["facts_extracted"] for r in results)
    total_failed = sum(r["docs_failed"] for r in results)
    errors = [r for r in results if r.get("error")]

    print("\n" + "=" * 60)
    print("PHASE 1 PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Companies processed : {processed}/{total} (skipped: {skipped})")
    print(f"Documents extracted : {total_docs}")
    print(f"Documents failed    : {total_failed}")
    print(f"Structured facts    : {total_facts}")
    if errors:
        print(f"\nErrors ({len(errors)} companies):")
        for e in errors:
            print(f"  - {e['company']}: {e['error']}")
    print("=" * 60 + "\n")


async def main_async(args: argparse.Namespace) -> None:
    # Step 1: Load GNEM Excel → PostgreSQL
    logger.info("Loading GNEM Excel into PostgreSQL...")
    companies_data = load_companies_from_excel()
    inserted, updated = sync_companies_to_db(companies_data)
    logger.info("GNEM sync: %d inserted, %d updated", inserted, updated)

    if args.load_only:
        logger.info("--load-only flag set. Stopping after DB sync.")
        return

    # Step 2: Get companies from DB (with IDs)
    all_companies = get_all_companies_from_db()
    logger.info("Loaded %d companies from DB", len(all_companies))

    # Filter if requested
    if args.company:
        target = args.company.lower()
        all_companies = [c for c in all_companies if target in c["company_name"].lower()]
        if not all_companies:
            logger.error("No company matching '%s'", args.company)
            sys.exit(1)
        logger.info("Filtered to %d matching company/ies", len(all_companies))

    if args.limit:
        all_companies = all_companies[:args.limit]
        logger.info("Limiting to first %d companies", args.limit)

    # Show credit estimate before running
    estimate = estimate_query_count(all_companies)
    logger.info(
        "Credit estimate: %d Tavily queries, ~%d credits (advanced depth)",
        estimate["total_queries"],
        estimate["estimated_tavily_credits"],
    )

    # Step 3: Run pipeline
    results = await run_pipeline(
        companies=all_companies,
        concurrency=args.concurrency,
        skip_if_has_docs=not args.rerun,
    )

    print_summary(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Georgia EV Intelligence — Phase 1 Pipeline")
    parser.add_argument(
        "--company", type=str, default=None,
        help="Run for a specific company name (partial match)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N companies",
    )
    parser.add_argument(
        "--load-only", action="store_true",
        help="Only load GNEM Excel to DB, skip search + extraction",
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-process companies even if they already have documents",
    )
    parser.add_argument(
        "--concurrency", type=int, default=COMPANY_CONCURRENCY,
        help=f"Parallel companies (default: {COMPANY_CONCURRENCY})",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
