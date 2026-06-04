"""Unit tests for the DedupCache (URL + content-hash deduplication)."""
import tempfile
from pathlib import Path

import pytest

from georgia_ev_intelligence.kb_builder.dedup import DedupCache


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_new_url_is_not_seen(tmp_dir: Path) -> None:
    with DedupCache(tmp_dir) as cache:
        assert not cache.is_url_seen("https://example.com/page")


def test_url_marked_seen(tmp_dir: Path) -> None:
    with DedupCache(tmp_dir) as cache:
        cache.mark_url_seen("https://example.com/page", "2026-01-01T00:00:00Z")
        assert cache.is_url_seen("https://example.com/page")


def test_hash_not_seen_initially(tmp_dir: Path) -> None:
    with DedupCache(tmp_dir) as cache:
        assert not cache.is_hash_seen("deadbeef" * 8)


def test_hash_marked_seen(tmp_dir: Path) -> None:
    with DedupCache(tmp_dir) as cache:
        h = "deadbeef" * 8
        cache.mark_hash_seen(h, "https://x.com", "2026-01-01T00:00:00Z")
        assert cache.is_hash_seen(h)


def test_is_duplicate_url_hit(tmp_dir: Path) -> None:
    with DedupCache(tmp_dir) as cache:
        url = "https://example.com/ev"
        cache.mark_url_seen(url, "2026-01-01T00:00:00Z")
        assert cache.is_duplicate(url, "somehash")


def test_is_duplicate_hash_hit(tmp_dir: Path) -> None:
    with DedupCache(tmp_dir) as cache:
        h = "aaaa" * 16
        cache.mark_hash_seen(h, "https://other.com", "2026-01-01T00:00:00Z")
        assert cache.is_duplicate("https://new-url.com", h)


def test_mark_seen_persists_across_instances(tmp_dir: Path) -> None:
    """State must survive closing and re-opening the cache."""
    url = "https://persist.test/page"
    h = "bbbb" * 16
    with DedupCache(tmp_dir) as c1:
        c1.mark_seen(url, h, "2026-01-01T00:00:00Z")

    with DedupCache(tmp_dir) as c2:
        assert c2.is_url_seen(url)
        assert c2.is_hash_seen(h)


def test_no_duplicate_if_nothing_seen(tmp_dir: Path) -> None:
    with DedupCache(tmp_dir) as cache:
        assert not cache.is_duplicate("https://fresh.com/page", "newhash123")
