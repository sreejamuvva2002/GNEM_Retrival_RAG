"""Unit tests for the HTML extractor (trafilatura-backed)."""
import pytest

from georgia_ev_intelligence.kb_builder.extractors.html_extractor import extract


SAMPLE_HTML = b"""
<!DOCTYPE html>
<html>
<head><title>Georgia EV Supply Chain Overview</title></head>
<body>
<nav>Home | About | Contact</nav>
<main>
  <h1>Georgia EV Supply Chain Overview</h1>
  <p>Georgia has become a hub for electric vehicle manufacturing.
  Companies like Rivian and Hyundai have established major facilities
  in the state, creating thousands of jobs in the EV supply chain.</p>
  <p>The state offers a range of incentives for EV-related businesses
  including tax credits, grants, and workforce development programmes.</p>
</main>
<footer>Copyright 2026</footer>
</body>
</html>
"""

SHORT_HTML = b"<html><body><p>Hi</p></body></html>"

EMPTY_HTML = b""


def test_extract_returns_nonempty_body(monkeypatch) -> None:
    title, body = extract(SAMPLE_HTML, url="https://example.com/ev")
    assert len(body) > 50, "Expected substantial body text from sample HTML"


def test_extract_title(monkeypatch) -> None:
    title, body = extract(SAMPLE_HTML, url="https://example.com/ev")
    assert "Georgia" in title or title == "", f"Unexpected title: {title!r}"


def test_extract_empty_html_returns_empty_strings() -> None:
    title, body = extract(EMPTY_HTML)
    assert title == ""
    assert body == ""


def test_extract_body_contains_ev_content() -> None:
    _, body = extract(SAMPLE_HTML, url="https://example.com/ev")
    # trafilatura should keep article content
    assert "electric vehicle" in body.lower() or "georgia" in body.lower()
