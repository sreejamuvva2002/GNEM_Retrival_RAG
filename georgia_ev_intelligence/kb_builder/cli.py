"""CLI entry-point for the kb_builder web crawler.

Usage
-----
One-shot crawl (all tiers):
    python -m georgia_ev_intelligence.kb_builder

One-shot crawl (company sites only, dry-run):
    python -m georgia_ev_intelligence.kb_builder --source company --dry-run

One-shot crawl without writing to PostgreSQL (JSONL only):
    python -m georgia_ev_intelligence.kb_builder --no-db

Periodic re-crawl (blocks until Ctrl-C):
    python -m georgia_ev_intelligence.kb_builder --schedule

Override crawl parameters inline:
    python -m georgia_ev_intelligence.kb_builder --depth 2 --concurrency 3 --limit 50
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _build_seeds(source: str) -> list[dict]:
    from georgia_ev_intelligence.kb_builder.seed_urls import (
        all_seeds,
        company_site_seeds,
        NEWS_SEEDS,
        GOV_SEEDS,
    )
    if source == "company":
        return company_site_seeds()
    if source == "news":
        return NEWS_SEEDS
    if source == "gov":
        return GOV_SEEDS
    return all_seeds()   # "all" (default)


def _make_crawl_fn(args: argparse.Namespace) -> callable:
    """Return a zero-argument callable suitable for the scheduler."""
    from georgia_ev_intelligence.kb_builder.crawler import run_crawl
    from georgia_ev_intelligence.shared import config

    def _crawl() -> None:
        seeds = _build_seeds(args.source)
        if args.limit:
            seeds = seeds[: args.limit]
        total = run_crawl(
            seeds,
            raw_docs_dir=config.RAW_DOCS_DIR,
            max_depth=args.depth,
            concurrency=args.concurrency,
            delay=args.delay,
            user_agent=config.CRAWLER_USER_AGENT,
            dry_run=args.dry_run,
            db=not args.no_db,
        )
        print(f"[crawl] Done. Documents written: {total}")

    return _crawl


def main() -> None:
    from georgia_ev_intelligence.shared import config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="GNEM Expanded KB — web crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["all", "company", "news", "gov"],
        default="all",
        help="Seed tier to crawl (default: all, priority A→B→C)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=config.CRAWLER_DEPTH,
        help=f"BFS depth per seed domain (default: {config.CRAWLER_DEPTH})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=config.CRAWLER_CONCURRENCY,
        help=f"Max parallel HTTP requests (default: {config.CRAWLER_CONCURRENCY})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=config.CRAWLER_DELAY_SECONDS,
        help=f"Min seconds between requests per domain (default: {config.CRAWLER_DELAY_SECONDS})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Process only the first N seed URLs (0 = unlimited, useful for smoke tests)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and extract but do NOT write documents (shows what would be written)",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Write to JSONL only; skip PostgreSQL upsert",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help=(
            "Block and run crawls periodically on "
            f"CRAWLER_SCHEDULE_CRON (default: {config.CRAWLER_SCHEDULE_CRON})"
        ),
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Create the raw_documents PostgreSQL table if it does not exist, then exit",
    )

    args = parser.parse_args()

    # --init-db: just ensure table exists and exit
    if args.init_db:
        from georgia_ev_intelligence.offline_pipeline.postgres_store import (
            ensure_raw_documents_table,
        )
        ensure_raw_documents_table()
        print("raw_documents table ensured in PostgreSQL.")
        return

    crawl_fn = _make_crawl_fn(args)

    if args.schedule:
        from georgia_ev_intelligence.kb_builder.scheduler import start
        start(crawl_fn, config.CRAWLER_SCHEDULE_CRON)
    else:
        crawl_fn()


if __name__ == "__main__":
    main()
