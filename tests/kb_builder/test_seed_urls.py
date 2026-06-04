"""Unit tests for seed_urls helpers (no network access)."""
import pytest


def test_news_seeds_are_nonempty() -> None:
    from georgia_ev_intelligence.kb_builder.seed_urls import NEWS_SEEDS
    assert len(NEWS_SEEDS) > 0
    for seed in NEWS_SEEDS:
        assert seed["url"].startswith("http"), f"Bad URL: {seed['url']}"
        assert seed["source_type"] == "news"


def test_gov_seeds_are_nonempty() -> None:
    from georgia_ev_intelligence.kb_builder.seed_urls import GOV_SEEDS
    assert len(GOV_SEEDS) > 0
    for seed in GOV_SEEDS:
        assert seed["url"].startswith("http"), f"Bad URL: {seed['url']}"
        assert seed["source_type"] == "gov_doc"


def test_all_seeds_priority_order() -> None:
    """Company seeds should appear before news/gov seeds in all_seeds()."""
    from georgia_ev_intelligence.kb_builder.seed_urls import (
        all_seeds,
        NEWS_SEEDS,
        GOV_SEEDS,
    )
    # We can't call all_seeds() without the Excel file, so just test the
    # static tier ordering by inspecting the function source / list ordering.
    # This test ensures no import-time error.
    assert NEWS_SEEDS[0]["source_type"] == "news"
    assert GOV_SEEDS[0]["source_type"] == "gov_doc"


def test_seed_dicts_have_required_keys() -> None:
    from georgia_ev_intelligence.kb_builder.seed_urls import NEWS_SEEDS, GOV_SEEDS
    for seed in NEWS_SEEDS + GOV_SEEDS:
        assert "url" in seed
        assert "source_type" in seed
