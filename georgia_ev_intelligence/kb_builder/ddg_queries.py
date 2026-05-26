"""Parse markdown files to extract DuckDuckGo queries and return seed URLs."""
from __future__ import annotations

import logging
import time
from typing import Optional

from ddgs import DDGS

logger = logging.getLogger(__name__)

# Mirror of the blocked-domain list in crawler.py so we filter bad URLs
# before they ever enter the crawl queue.
_BLOCKED_DOMAINS: frozenset[str] = frozenset({
    "chamberofcommerce.com",
    "yelp.com",
    "yellowpages.com",
    "manta.com",
    "dnb.com",
    "zoominfo.com",
    "bloomberg.com",
    "hoovers.com",
    "bbb.org",
    "corporationwiki.com",
    "opencorporates.com",
    "bizapedia.com",
    "bizstanding.com",
    "corporateregistration.com",
})


def _is_blocked(url: str) -> bool:
    import urllib.parse
    domain = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
    return domain in _BLOCKED_DOMAINS


def parse_queries(md_file_path: str) -> list[tuple[str, str]]:
    """Parse the markdown file and return a list of (company_name, query)."""
    queries = []
    current_company: Optional[str] = None

    with open(md_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("### "):
                current_company = line[4:].strip()
            elif line.startswith("- "):
                query = line[2:].strip()
                if query and current_company:
                    queries.append((current_company, query))
                elif query:
                    queries.append(("Unknown", query))

    return queries


def _ddg_search(ddgs: DDGS, query: str, max_results: int) -> list[dict]:
    """Try html backend first, fall back to auto if it returns nothing."""
    try:
        results = list(ddgs.text(query, backend="html", max_results=max_results))
        if results:
            return results
    except Exception as exc:
        logger.debug("DDG html backend failed for %r: %s", query, exc)

    # Fallback: let ddgs choose the best available backend
    try:
        results = list(ddgs.text(query, max_results=max_results))
        return results or []
    except Exception as exc:
        logger.warning("DDG fallback also failed for %r: %s", query, exc)
        return []


def get_ddg_seeds(
    md_file_path: str,
    max_results_per_query: int = 3,
    max_queries: int = 0,
) -> list[dict]:
    """Execute DDG queries and return seed dicts, filtering blocked domains."""
    queries = parse_queries(md_file_path)
    if max_queries > 0:
        queries = queries[:max_queries]

    logger.info("Loaded %d queries from %s", len(queries), md_file_path)

    seeds: list[dict] = []
    ddgs = DDGS()

    for company, query in queries:
        logger.info("Running DDG query for %s: %s", company, query)
        try:
            results = _ddg_search(ddgs, query, max_results=max_results_per_query)
            accepted = 0
            for res in results:
                url = res.get("href", "").strip()
                if not url:
                    continue
                if _is_blocked(url):
                    logger.debug("DDG result filtered (blocked domain): %s", url)
                    continue
                seeds.append({
                    "url": url,
                    "source_type": "ddg_search",
                    "linked_company_id": company,
                })
                accepted += 1
            logger.info(
                "  → %d result(s) accepted, %d filtered for %s",
                accepted, len(results) - accepted, company,
            )
            # Respectful delay between DDG requests
            time.sleep(1.0)
        except Exception as exc:
            logger.error("Error executing DDG search for %r: %s", query, exc)

    logger.info("Generated %d seed URLs from DDG searches.", len(seeds))
    return seeds
