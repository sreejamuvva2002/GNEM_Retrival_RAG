"""APScheduler-based periodic re-crawl scheduler.

Usage (standalone):
    python -m georgia_ev_intelligence.kb_builder --schedule

This keeps the process alive and triggers a full crawl on the configured
CRAWLER_SCHEDULE_CRON expression (default: every Sunday at 02:00).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def start(crawl_fn, cron_expr: str) -> None:
    """Block forever, running *crawl_fn()* on *cron_expr* schedule.

    Parameters
    ----------
    crawl_fn:
        Zero-argument callable that executes the crawl (typically a lambda
        wrapping ``crawler.run_crawl``).
    cron_expr:
        A 5-field cron expression, e.g. ``"0 2 * * 0"`` for every Sunday
        at 02:00.
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler  # type: ignore
        from apscheduler.triggers.cron import CronTrigger  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "apscheduler is required: pip install apscheduler>=3.10"
        ) from exc

    minute, hour, day, month, day_of_week = cron_expr.split()

    scheduler = BlockingScheduler(timezone="America/New_York")
    scheduler.add_job(
        crawl_fn,
        CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
        ),
        id="gnem_crawl",
        name="GNEM Expanded KB Crawl",
        misfire_grace_time=3600,
    )

    logger.info(
        "Scheduler started. Next run on cron: %s (America/New_York)", cron_expr
    )
    print(f"[scheduler] Periodic crawl active — cron: {cron_expr}  (Ctrl-C to stop)")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user.")
