"""Async BFS web crawler with robots.txt compliance and per-domain rate-limiting.

Design:
- httpx AsyncClient for concurrent fetching (configurable pool size).
- asyncio.Semaphore limits total in-flight requests.
- Per-domain asyncio.Lock + last-fetch timestamp enforces minimum delay.
- urllib.robotparser checks robots.txt before every fetch.
- BFS queue bounded by CRAWLER_DEPTH.
- Dispatches to the correct extractor based on Content-Type / URL extension.
- Writes every accepted document via writer.write_document().
"""
from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content-type / extension helpers
# ---------------------------------------------------------------------------

_PDF_RE  = re.compile(r"\.pdf(\?.*)?$", re.IGNORECASE)
_DOCX_RE = re.compile(r"\.(docx?)(\?.*)?$", re.IGNORECASE)
_IMAGE_RE = re.compile(r"\.(jpg|jpeg|png|gif|svg|webp|ico)(\?.*)?$", re.IGNORECASE)
_EXCEL_RE = re.compile(r"\.(xlsx?|xlsm)(\?.*)?$", re.IGNORECASE)
_CSV_RE = re.compile(r"\.(csv|tsv)(\?.*)?$", re.IGNORECASE)
_JSON_RE = re.compile(r"\.json(\?.*)?$", re.IGNORECASE)
_XML_RE = re.compile(r"\.xml(\?.*)?$", re.IGNORECASE)
_TEXT_RE = re.compile(r"\.(txt|md)(\?.*)?$", re.IGNORECASE)

_SKIP_EXTS = re.compile(
    r"\.(css|js|woff2?|ttf|eot|mp4|mp3|zip|tar|gz)$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Aggregator / directory sites that reliably 403 or contain thin content
# Add domains here to permanently skip them for DDG seeds.
# ---------------------------------------------------------------------------
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


def _guess_file_type(url: str, content_type: str) -> str:
    ct = content_type.lower()
    if "pdf" in ct or _PDF_RE.search(url):
        return "pdf"
    if "officedocument" in ct or "msword" in ct or _DOCX_RE.search(url):
        return "docx"
    if "image" in ct or _IMAGE_RE.search(url):
        return "image"
    if "excel" in ct or "spreadsheet" in ct or _EXCEL_RE.search(url):
        return "excel"
    if "csv" in ct or "tab-separated" in ct or _CSV_RE.search(url):
        return "csv"
    if "json" in ct or _JSON_RE.search(url):
        return "json"
    if "xml" in ct or _XML_RE.search(url):
        return "xml"
    if "text/plain" in ct or "markdown" in ct or _TEXT_RE.search(url):
        return "text"
    return "html"


def _is_skippable(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if bool(_SKIP_EXTS.search(parsed.path)):
        return True
    # Strip leading www. for domain matching
    domain = parsed.netloc.lower().lstrip("www.")
    if domain in _BLOCKED_DOMAINS:
        logger.debug("Blocked domain — skipping %s", url)
        return True
    return False


def _same_domain(base: str, target: str) -> bool:
    b = urllib.parse.urlparse(base).netloc.lstrip("www.")
    t = urllib.parse.urlparse(target).netloc.lstrip("www.")
    return b == t


def _extract_links(html: str, base_url: str) -> list[str]:
    """Pull all <a href> links from raw HTML text."""
    pattern = re.compile(r'href=["\']([^"\'#\s]+)["\']', re.IGNORECASE)
    links: list[str] = []
    for m in pattern.finditer(html):
        href = m.group(1)
        resolved = urllib.parse.urljoin(base_url, href)
        parsed = urllib.parse.urlparse(resolved)
        if parsed.scheme not in ("http", "https"):
            continue
        # Strip fragment
        clean = urllib.parse.urlunparse(parsed._replace(fragment=""))
        links.append(clean)
    return links


# ---------------------------------------------------------------------------
# Per-domain politeness tracker
# ---------------------------------------------------------------------------

class _DomainThrottle:
    """Ensures at least `delay` seconds between requests to the same domain."""

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}

    def _key(self, url: str) -> str:
        return urllib.parse.urlparse(url).netloc

    async def acquire(self, url: str) -> None:
        key = self._key(url)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        async with self._locks[key]:
            now = asyncio.get_event_loop().time()
            last = self._last.get(key, 0.0)
            wait = self._delay - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last[key] = asyncio.get_event_loop().time()


# ---------------------------------------------------------------------------
# Robots.txt cache
# ---------------------------------------------------------------------------

class _RobotsCache:
    """Fetches and caches robots.txt per domain."""

    def __init__(self, user_agent: str) -> None:
        self._ua = user_agent
        self._cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    async def can_fetch(self, url: str, client) -> bool:
        parsed = urllib.parse.urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain not in self._cache:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = f"{domain}/robots.txt"
            try:
                resp = await client.get(robots_url, timeout=5)
                rp.parse(resp.text.splitlines())
            except Exception:
                # If robots.txt is unreachable, allow crawling
                rp.allow_all = True
            self._cache[domain] = rp
        return self._cache[domain].can_fetch(self._ua, url)


# ---------------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------------

async def crawl(
    seeds: list[dict],
    *,
    raw_docs_dir: Path,
    max_depth: int,
    concurrency: int,
    delay: float,
    user_agent: str,
    dry_run: bool = False,
    db: bool = True,
    b2: bool = True,
) -> int:
    """Crawl all seed URLs with BFS up to max_depth.

    Returns the number of documents successfully written.
    """
    try:
        import httpx  # type: ignore
    except ImportError as exc:
        raise ImportError("httpx is required: pip install httpx[http2]>=0.27") from exc

    from georgia_ev_intelligence.kb_builder.dedup import DedupCache
    from georgia_ev_intelligence.kb_builder.models import RawDocument
    from georgia_ev_intelligence.kb_builder.writer import write_document
    from georgia_ev_intelligence.kb_builder.extractors import (
        html_extractor,
        pdf_extractor,
        docx_extractor,
        excel_extractor,
        csv_extractor,
        json_extractor,
        xml_extractor,
        text_extractor,
    )

    sem = asyncio.Semaphore(concurrency)
    throttle = _DomainThrottle(delay)

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/pdf,application/xhtml+xml,*/*",
    }

    written = 0

    with DedupCache(raw_docs_dir) as dedup:
        robots = _RobotsCache(user_agent)

        # BFS queue: (url, depth, source_type, linked_company_id, seed_url)
        queue: asyncio.Queue = asyncio.Queue()
        for seed in seeds:
            queue.put_nowait((seed["url"], 0, seed["source_type"],
                              seed.get("linked_company_id"), seed["url"]))

        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(15.0, connect=10.0),
            http2=True,
            verify=False,       # some company sites have bad certs
        ) as client:

            async def fetch_and_store(
                url: str,
                depth: int,
                source_type: str,
                linked_company_id: Optional[str],
                seed_url: str,
            ) -> list[tuple]:
                """Fetch one URL, extract text, persist, return child links."""
                nonlocal written
                child_items: list[tuple] = []

                if _is_skippable(url):
                    return child_items

                if dedup.is_url_seen(url):
                    return child_items

                allowed = await robots.can_fetch(url, client)
                if not allowed:
                    logger.info("robots.txt disallows %s — skipping", url)
                    return child_items

                await throttle.acquire(url)

                async with sem:
                    try:
                        resp = await client.get(url)
                        status = resp.status_code
                    except Exception as exc:
                        logger.warning("Fetch error %s: %s", url, exc)
                        dedup.mark_url_seen(url, datetime.now(timezone.utc).isoformat())
                        return child_items

                # Skip non-200 responses immediately — don't try to extract
                # from error pages (403, 404, 429, 5xx, etc.)
                if status != 200:
                    logger.info("Skipping %s — HTTP %d", url, status)
                    dedup.mark_url_seen(url, datetime.now(timezone.utc).isoformat())
                    return child_items

                crawled_at = datetime.now(timezone.utc)
                ct = resp.headers.get("content-type", "text/html")
                file_type = _guess_file_type(url, ct)

                # Extract
                title, body = "", ""
                raw_bytes = resp.content
                if file_type == "pdf":
                    title, body = pdf_extractor.extract(raw_bytes)
                elif file_type == "docx":
                    title, body = docx_extractor.extract(raw_bytes)
                elif file_type == "excel":
                    title, body = excel_extractor.extract(raw_bytes)
                elif file_type == "csv":
                    title, body = csv_extractor.extract(raw_bytes)
                elif file_type == "json":
                    title, body = json_extractor.extract(raw_bytes)
                elif file_type == "xml":
                    title, body = xml_extractor.extract(raw_bytes)
                elif file_type == "text":
                    title, body = text_extractor.extract(raw_bytes)
                elif file_type == "image":
                    title = url.split("/")[-1] or "image"
                    body = f"Image file extracted from {url}"
                else:
                    title, body = html_extractor.extract(raw_bytes, url=url)

                # Enqueue child links (HTML only, same-domain, within depth)
                if file_type == "html" and depth < max_depth:
                    html_str = raw_bytes.decode("utf-8", errors="replace")
                    for link in _extract_links(html_str, url):
                        if _same_domain(seed_url, link) and not dedup.is_url_seen(link):
                            child_items.append(
                                (link, depth + 1, source_type,
                                 linked_company_id, seed_url)
                            )

                if not body or len(body) < 100:
                    logger.info(
                        "Skipping %s — body too short (%d chars) after extraction",
                        url, len(body) if body else 0,
                    )
                    dedup.mark_url_seen(url, crawled_at.isoformat())
                    return child_items

                # Content-hash dedup
                import hashlib
                norm = body.strip().lower()
                content_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()

                if dedup.is_hash_seen(content_hash):
                    logger.info("Skipping %s — duplicate content hash", url)
                    dedup.mark_url_seen(url, crawled_at.isoformat())
                    return child_items

                domain = urllib.parse.urlparse(url).netloc

                doc = RawDocument(
                    url=url,
                    body_text=body,
                    source_type=source_type,
                    title=title,
                    domain=domain,
                    http_status=status,
                    linked_company_id=linked_company_id,
                    crawled_at=crawled_at,
                    file_type=file_type,
                    raw_binary=raw_bytes,   # always stored: HTML for B2, PDF/DOCX too
                )

                dedup.mark_seen(url, content_hash, crawled_at.isoformat())

                if not dry_run:
                    write_document(doc, raw_docs_dir, db=db, b2=b2)
                    written += 1
                else:
                    logger.info("[DRY-RUN] Would write: %s (%d chars)", url, len(body))
                    written += 1

                return child_items

            # BFS loop — drain queue with async workers
            active: set[asyncio.Task] = set()

            while not queue.empty() or active:
                # Launch tasks up to concurrency limit
                while not queue.empty() and len(active) < concurrency * 2:
                    item = await queue.get()
                    task = asyncio.create_task(fetch_and_store(*item))
                    active.add(task)

                if not active:
                    break

                done, active = await asyncio.wait(
                    active, return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    try:
                        children = t.result()
                        for child in children:
                            queue.put_nowait(child)
                    except Exception as exc:
                        logger.error("Task error: %s", exc)

    logger.info("Crawl complete. Documents written: %d", written)
    return written


def run_crawl(seeds: list[dict], **kwargs) -> int:
    """Synchronous wrapper around the async crawl coroutine."""
    return asyncio.run(crawl(seeds, **kwargs))
