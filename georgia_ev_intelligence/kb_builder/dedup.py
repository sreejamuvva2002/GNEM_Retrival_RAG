"""Two-layer deduplication: URL-level and content-hash-level.

Both layers use a lightweight SQLite database stored in kb/raw_docs/.dedup.db
so state persists across runs (required for the periodic re-crawl cadence).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


_DB_NAME = ".dedup.db"


def _db_path(raw_docs_dir: Path) -> Path:
    return raw_docs_dir / _DB_NAME


def _get_conn(raw_docs_dir: Path) -> sqlite3.Connection:
    db = _db_path(raw_docs_dir)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_urls (
            url TEXT PRIMARY KEY,
            crawled_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_hashes (
            content_hash TEXT PRIMARY KEY,
            first_url TEXT,
            crawled_at TEXT
        )
    """)
    conn.commit()
    return conn


class DedupCache:
    """Persistent URL + content-hash deduplication cache backed by SQLite."""

    def __init__(self, raw_docs_dir: Path) -> None:
        self._raw_docs_dir = raw_docs_dir
        self._conn = _get_conn(raw_docs_dir)

    # ------------------------------------------------------------------
    # URL layer
    # ------------------------------------------------------------------

    def is_url_seen(self, url: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM seen_urls WHERE url = ?", (url,)
        )
        return cur.fetchone() is not None

    def mark_url_seen(self, url: str, crawled_at: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO seen_urls (url, crawled_at) VALUES (?, ?)",
            (url, crawled_at),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Content-hash layer
    # ------------------------------------------------------------------

    def is_hash_seen(self, content_hash: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM seen_hashes WHERE content_hash = ?", (content_hash,)
        )
        return cur.fetchone() is not None

    def mark_hash_seen(self, content_hash: str, url: str, crawled_at: str) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO seen_hashes
               (content_hash, first_url, crawled_at) VALUES (?, ?, ?)""",
            (content_hash, url, crawled_at),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Combined check
    # ------------------------------------------------------------------

    def is_duplicate(self, url: str, content_hash: str) -> bool:
        """Return True if URL or content-hash has been seen before."""
        return self.is_url_seen(url) or self.is_hash_seen(content_hash)

    def mark_seen(self, url: str, content_hash: str, crawled_at: str) -> None:
        """Record both the URL and content hash as seen."""
        self.mark_url_seen(url, crawled_at)
        self.mark_hash_seen(content_hash, url, crawled_at)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DedupCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
