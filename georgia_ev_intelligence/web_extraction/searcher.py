"""
Phase 1 — Searcher
Uses Tavily Search API to find documents for each company.
Replaces DuckDuckGo entirely.

Tavily search_depth="advanced" = 2 API credits per call.
Returns structured results ready for the extractor.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from shared.config import Config
from shared.logger import get_logger

logger = get_logger("web_extraction.searcher")

# Domains known to have high-quality EV supply chain content
# Tavily will still search all domains, but we log when these appear
PRIORITY_DOMAINS = {
    "georgia.org", "selectgeorgia.com", "savannahjda.com",
    "sec.gov", "energy.gov", "epd.georgia.gov",
    "gaports.com", "hmgma.com", "kiageorgia.com",
    "skon.co", "autonews.com", "emobility.uga.edu",
    "reuters.com", "bloomberg.com",
}

# Domains to exclude from extraction (login walls, paywalls with no value)
BLOCKLIST_DOMAINS = {
    "linkedin.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "youtube.com", "tiktok.com",
    "indeed.com", "glassdoor.com", "ziprecruiter.com",
    "maps.google.com", "google.com/maps",
}


def _is_blocked(url: str) -> bool:
    """Check if a URL should be skipped."""
    url_lower = url.lower()
    return any(domain in url_lower for domain in BLOCKLIST_DOMAINS)


def _is_priority(url: str) -> bool:
    return any(domain in url.lower() for domain in PRIORITY_DOMAINS)


async def tavily_search(
    query: str,
    max_results: int = 10,
    search_depth: str = "advanced",
) -> list[dict[str, Any]]:
    """
    Call Tavily Search API.

    Args:
        query: Search query string
        max_results: How many results to return (max 20)
        search_depth: "basic" (1 credit) or "advanced" (2 credits)

    Returns:
        List of result dicts: {url, title, content(snippet), score}
    """
    cfg = Config.get()
    payload = {
        "api_key": cfg.tavily_api_key,
        "query": query,
        "max_results": min(max_results, 20),
        "search_depth": search_depth,
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,  # Snippets only — full content via Extract
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post("https://api.tavily.com/search", json=payload)
        response.raise_for_status()
        data = response.json()

    raw_results = data.get("results", [])
    results = []
    for item in raw_results:
        url = item.get("url", "")
        if not url or _is_blocked(url):
            continue
        results.append({
            "url": url,
            "title": item.get("title", ""),
            "snippet": item.get("content", ""),   # Tavily calls it "content" but it's a snippet
            "score": float(item.get("score", 0.0)),
            "is_priority": _is_priority(url),
        })

    logger.debug("Tavily search '%s' → %d results", query[:60], len(results))
    return results


async def search_company(
    company: dict[str, Any],
    queries: list[dict[str, Any]],
    concurrency: int = 3,
    delay_between_batches: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Run all queries for a single company and collect unique URLs.

    Args:
        company: Company dict from DB
        queries: List of query dicts from query_generator.build_queries()
        concurrency: Parallel Tavily requests at once (3 is safe)
        delay_between_batches: Seconds between batches (rate limit courtesy)

    Returns:
        Deduplicated list of URL result dicts with company metadata attached
    """
    company_name = company.get("company_name", "")
    logger.info("Searching for %s (%d queries)", company_name, len(queries))

    seen_urls: set[str] = set()
    all_results: list[dict[str, Any]] = []

    semaphore = asyncio.Semaphore(concurrency)

    async def _run_query(q: dict[str, Any]) -> list[dict[str, Any]]:
        async with semaphore:
            try:
                results = await tavily_search(
                    query=q["query_text"],
                    max_results=q.get("max_results", 10),
                    search_depth=q.get("search_depth", "advanced"),
                )
                return results
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    logger.warning("Tavily rate limit hit — sleeping 10s")
                    await asyncio.sleep(10.0)
                else:
                    logger.warning("Tavily search failed [%d] for '%s'", exc.response.status_code, q["query_text"][:50])
                return []
            except Exception as exc:
                logger.warning("Tavily search error for '%s': %s", q["query_text"][:50], exc)
                return []

    # Run in batches to be respectful of rate limits
    batch_size = concurrency * 2
    for batch_start in range(0, len(queries), batch_size):
        batch = queries[batch_start: batch_start + batch_size]
        tasks = [_run_query(q) for q in batch]
        batch_results = await asyncio.gather(*tasks)

        for query_results in batch_results:
            for result in query_results:
                url = result["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    result["company_id"] = company.get("id")
                    result["company_name"] = company_name
                    all_results.append(result)

        if batch_start + batch_size < len(queries):
            await asyncio.sleep(delay_between_batches)

    # Sort: priority domains first, then by Tavily score
    all_results.sort(key=lambda r: (-int(r["is_priority"]), -r["score"]))
    logger.info(
        "Company '%s': %d unique URLs found (%d priority)",
        company_name,
        len(all_results),
        sum(1 for r in all_results if r["is_priority"]),
    )
    return all_results
