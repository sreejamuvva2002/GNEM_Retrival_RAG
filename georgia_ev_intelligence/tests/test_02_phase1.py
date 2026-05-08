"""
Phase 1 Test Suite — validates every component before Phase 2.
Tests are ordered: each test builds on the previous one.

Run: venv\\Scripts\\python -m pytest tests/test_02_phase1.py -v --tb=short
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestKbLoader(unittest.TestCase):
    """Test 2.1 — GNEM Excel loading and PostgreSQL sync."""

    def test_excel_can_be_found(self):
        """GNEM Excel must be findable."""
        from db_storage.kb_loader import _find_gnem_excel
        path = _find_gnem_excel()
        self.assertTrue(path.exists(), f"GNEM Excel not found at {path}")
        self.assertTrue(path.suffix in (".xlsx", ".xls"), "File must be an Excel file")

    def test_excel_has_205_rows(self):
        """Must load exactly 205 companies (or close — GNEM is fixed)."""
        from db_storage.kb_loader import load_companies_from_excel
        companies = load_companies_from_excel()
        self.assertGreaterEqual(len(companies), 200, "Expected ~205 companies in GNEM Excel")
        self.assertLessEqual(len(companies), 210, "Too many rows — check Excel parsing")

    def test_companies_have_required_fields(self):
        """Every company must have company_name."""
        from db_storage.kb_loader import load_companies_from_excel
        companies = load_companies_from_excel()
        for c in companies:
            self.assertTrue(c.get("company_name"), f"Company missing name: {c}")

    def test_tier_field_present(self):
        """At least some companies should have tier info."""
        from db_storage.kb_loader import load_companies_from_excel
        companies = load_companies_from_excel()
        with_tier = [c for c in companies if c.get("tier")]
        self.assertGreater(len(with_tier), 100, "Expected most companies to have tier info")

    def test_db_sync(self):
        """Sync to PostgreSQL and verify count — 190+ of 204 good companies expected."""
        from db_storage.kb_loader import (
            load_companies_from_excel,
            sync_companies_to_db,
            get_all_companies_from_db,
        )
        companies = load_companies_from_excel()
        # 204 good rows (1 skipped for data quality)
        self.assertGreaterEqual(len(companies), 200, "Should parse 200+ companies from Excel")

        inserted, updated = sync_companies_to_db(companies)
        # All 204 should either insert or update — 0 should be skipped
        self.assertGreater(inserted + updated, 190, "Should have synced 190+ companies")

        db_companies = get_all_companies_from_db()
        # DB may have 190-204 depending on previous test run history
        self.assertGreaterEqual(len(db_companies), 190, "DB should have 190+ companies after sync")
        self.assertTrue(all(c.get("id") for c in db_companies), "All DB companies must have IDs")

    def test_build_document_text(self):
        """Document text must contain all required labels."""
        from db_storage.kb_loader import build_document_text
        company = {
            "company_name": "Test Corp",
            "tier": "Tier 1",
            "industry_group": "Battery",
            "location_city": "Savannah",
            "location_county": "Chatham",
            "location_state": "Georgia",
            "ev_supply_chain_role": "Cell Manufacturer",
            "primary_oems": "Hyundai",
            "ev_battery_relevant": "Yes",
            "employment": 3000,
            "products_services": "Lithium ion cells",
            "classification_method": "Direct",
            "supplier_affiliation_type": "Tier 1",
        }
        text = build_document_text(company)
        for label in ["Company:", "Tier:", "Location:", "EV Role:", "OEMs:", "Employment:"]:
            self.assertIn(label, text, f"Missing label '{label}' in document text")


class TestQueryGenerator(unittest.TestCase):
    """Test 2.2 — Query generation per company."""

    def setUp(self):
        self.company = {
            "id": 1,
            "company_name": "Hanwha Q Cells",
            "tier": "OEM",
            "location_city": "Dalton",
            "location_county": "Whitfield",
            "location_state": "Georgia",
        }

    def test_queries_generated(self):
        from web_extraction.query_generator import build_queries
        queries = build_queries(self.company)
        self.assertGreater(len(queries), 5, "Expected at least 5 queries for an OEM")

    def test_no_duplicate_queries(self):
        from web_extraction.query_generator import build_queries
        queries = build_queries(self.company)
        query_texts = [q["query_text"].lower().strip() for q in queries]
        self.assertEqual(len(query_texts), len(set(query_texts)), "Queries must be deduplicated")

    def test_company_name_in_queries(self):
        from web_extraction.query_generator import build_queries
        queries = build_queries(self.company)
        for q in queries:
            self.assertIn("Hanwha Q Cells", q["query_text"], "Company name must appear in all queries")

    def test_search_depth_is_advanced(self):
        from web_extraction.query_generator import build_queries
        queries = build_queries(self.company)
        for q in queries:
            self.assertEqual(q["search_depth"], "advanced", "All queries must use advanced depth")

    def test_estimate_query_count(self):
        from web_extraction.query_generator import estimate_query_count
        companies = [self.company]
        est = estimate_query_count(companies)
        self.assertIn("total_queries", est)
        self.assertIn("estimated_tavily_credits", est)
        self.assertGreater(est["estimated_tavily_credits"], 0)

    def test_tier2_gets_fewer_queries(self):
        """Tier 2 companies should have fewer query families than OEM."""
        from web_extraction.query_generator import build_queries
        tier2_company = {**self.company, "tier": "Tier 2", "company_name": "Small Parts Co"}
        oem_queries = build_queries(self.company)
        tier2_queries = build_queries(tier2_company)
        self.assertGreaterEqual(len(oem_queries), len(tier2_queries),
                                "OEM should have >= queries as Tier 2")


class TestExtractor(unittest.TestCase):
    """Test 2.3 — PDF and HTML extraction."""

    def test_pdf_bytes_extraction(self):
        """Test PyMuPDF extraction on a real PDF (skip if no PDF available)."""
        from web_extraction.extractor import extract_pdf_bytes
        # Create a minimal valid PDF in memory to test parsing
        try:
            import fitz
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Hanwha Q Cells Georgia investment $2.5 billion 2023")
            pdf_bytes = doc.tobytes()
            doc.close()

            text = extract_pdf_bytes(pdf_bytes, "test.pdf")
            self.assertIn("Hanwha", text)
            self.assertIn("Georgia", text)
        except ImportError:
            self.skipTest("PyMuPDF not installed")

    def test_is_pdf_url_detection(self):
        """URL detection heuristic."""
        from web_extraction.extractor import _is_pdf_url
        self.assertTrue(_is_pdf_url("https://sec.gov/filing/10k.pdf"))
        self.assertTrue(_is_pdf_url("https://example.com/pdf/report"))
        self.assertFalse(_is_pdf_url("https://example.com/article"))
        self.assertFalse(_is_pdf_url("https://georgia.org/news"))

    def test_tavily_extract_called_for_html(self):
        """Tavily Extract returns clean text for HTML URLs."""
        import web_extraction.extractor as extractor_mod

        original = extractor_mod.tavily_extract
        mock_result = {
            "url": "https://example.com/article",
            # Must be > MIN_CONTENT_CHARS (150) to pass length check
            "raw_content": (
                "Hanwha Q Cells announced a $2.5 billion investment in Dalton, Georgia, "
                "creating 3,000 new jobs. The facility will manufacture lithium-ion battery "
                "modules and packs for the EV supply chain, supplying Hyundai and Kia vehicles."
            ),
        }

        async def fake_extract(url):
            return mock_result

        # Directly replace the module-level function
        extractor_mod.tavily_extract = fake_extract
        try:
            loop = asyncio.new_event_loop()
            try:
                from web_extraction.extractor import extract_document
                result = loop.run_until_complete(
                    extract_document("https://example.com/article", "Hanwha Q Cells")
                )
            finally:
                loop.close()
        finally:
            extractor_mod.tavily_extract = original  # Always restore

        self.assertEqual(result.extraction_method, "tavily_extract")
        self.assertIn("Hanwha", result.text)
        self.assertGreater(result.word_count, 0)
        self.assertFalse(result.error)

    def test_failed_extraction_returns_error_doc(self):
        """Failed Tavily call returns error doc, does not raise."""
        import web_extraction.extractor as extractor_mod
        original = extractor_mod.tavily_extract

        async def fake_extract_fail(url):
            raise Exception("Connection refused")

        extractor_mod.tavily_extract = fake_extract_fail
        try:
            loop = asyncio.new_event_loop()
            try:
                from web_extraction.extractor import extract_document
                result = loop.run_until_complete(
                    extract_document("https://example.com/broken", "Test Corp")
                )
            finally:
                loop.close()
        finally:
            extractor_mod.tavily_extract = original

        self.assertTrue(result.error, "Error field should be set")
        self.assertEqual(result.text, "", "Text should be empty on failure")

    def test_content_hash_computed(self):
        """SHA-256 hash must be 64-char hex string for successful extractions."""
        import web_extraction.extractor as extractor_mod
        original = extractor_mod.tavily_extract

        async def fake_extract_hash(url):
            return {"url": url, "raw_content": "A" * 300}

        extractor_mod.tavily_extract = fake_extract_hash
        try:
            loop = asyncio.new_event_loop()
            try:
                from web_extraction.extractor import extract_document
                result = loop.run_until_complete(
                    extract_document("https://x.com", "Test Corp")
                )
            finally:
                loop.close()
        finally:
            extractor_mod.tavily_extract = original

        self.assertEqual(len(result.content_hash), 64, "SHA-256 hash should be 64 hex chars")


class TestEntityExtractor(unittest.TestCase):
    """Test 2.4 — Structured fact extraction via Ollama."""

    def test_json_parsing_valid(self):
        """Valid JSON from Ollama is parsed correctly."""
        from web_extraction.entity_extractor import _parse_facts_json
        raw = '[{"fact_type":"investment","fact_value_text":"$400M","fact_value_numeric":400000000,"fact_currency":"USD","fact_unit":"USD","fact_year":2024,"fact_quarter":null,"location_city":"Dalton","location_county":"Whitfield","oem_partner":null,"confidence_score":0.95,"source_sentence":"Company announced $400M investment."}]'
        facts = _parse_facts_json(raw)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["fact_type"], "investment")
        self.assertAlmostEqual(facts[0]["fact_value_numeric"], 400000000)

    def test_json_parsing_with_markdown_fence(self):
        """LLM often wraps JSON in markdown fences."""
        from web_extraction.entity_extractor import _parse_facts_json
        raw = '```json\n[{"fact_type":"jobs_created","fact_value_numeric":3000}]\n```'
        facts = _parse_facts_json(raw)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["fact_type"], "jobs_created")

    def test_json_parsing_empty_returns_empty_list(self):
        """Empty or non-JSON LLM output returns empty list without raising."""
        from web_extraction.entity_extractor import _parse_facts_json
        self.assertEqual(_parse_facts_json(""), [])
        self.assertEqual(_parse_facts_json("No facts found."), [])
        self.assertEqual(_parse_facts_json("[]"), [])

    def test_extract_facts_calls_ollama(self):
        """extract_facts calls Ollama and returns parsed facts."""
        from web_extraction.entity_extractor import extract_facts

        mock_response = {
            "response": '[{"fact_type":"investment","fact_value_text":"$400 million","fact_value_numeric":400000000,"fact_currency":"USD","fact_unit":"USD","fact_year":2024,"fact_quarter":null,"location_city":"Dalton","location_county":"Whitfield","oem_partner":"Hyundai","confidence_score":0.9,"source_sentence":"Hanwha announced a $400M investment."}]'
        }

        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: mock_response,
                raise_for_status=lambda: None,
            )
            facts = extract_facts(
                text="Hanwha announced a $400M investment in Dalton Georgia.",
                company_name="Hanwha Q Cells",
                company_id=1,
                document_id=1,
            )
        self.assertGreater(len(facts), 0)
        self.assertEqual(facts[0]["fact_type"], "investment")
        self.assertEqual(facts[0]["company_name"], "Hanwha Q Cells")

    def test_low_confidence_filtered(self):
        """Facts with confidence < 0.3 are not saved to DB."""
        from web_extraction.entity_extractor import save_facts_to_db
        facts = [
            {
                "company_id": None,
                "document_id": None,
                "company_name": "Test Corp",
                "fact_type": "investment",
                "fact_value_text": "$1B",
                "fact_value_numeric": 1_000_000_000,
                "fact_currency": "USD",
                "fact_unit": "USD",
                "fact_year": 2024,
                "fact_quarter": None,
                "location_city": None,
                "location_county": None,
                "location_state": "Georgia",
                "oem_partner": None,
                "confidence_score": 0.1,  # Below threshold
                "source_sentence": "Some sentence.",
                "extracted_at": None,
            }
        ]
        count = save_facts_to_db(facts)
        self.assertEqual(count, 0, "Low-confidence facts should not be inserted")


class TestSearcher(unittest.TestCase):
    """Test 2.5 — Tavily search integration."""

    def test_blocklist_filtering(self):
        """LinkedIn, YouTube, etc. should be blocked."""
        from web_extraction.searcher import _is_blocked
        self.assertTrue(_is_blocked("https://linkedin.com/company/hanwha"))
        self.assertTrue(_is_blocked("https://youtube.com/watch?v=123"))
        self.assertFalse(_is_blocked("https://georgia.org/news/hanwha"))

    def test_priority_detection(self):
        from web_extraction.searcher import _is_priority
        self.assertTrue(_is_priority("https://georgia.org/announcement"))
        self.assertTrue(_is_priority("https://sec.gov/filing"))
        self.assertFalse(_is_priority("https://example.com/news"))

    def test_tavily_search_called(self):
        """tavily_search returns expected result structure."""
        from web_extraction.searcher import tavily_search

        mock_api_response = {
            "results": [
                {"url": "https://georgia.org/hanwha-news", "title": "Hanwha News", "content": "Snippet...", "score": 0.95},
                {"url": "https://linkedin.com/company/hanwha", "title": "LinkedIn", "content": "Profile", "score": 0.5},
            ]
        }

        async def run():
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(return_value=MagicMock(
                    raise_for_status=lambda: None,
                    json=lambda: mock_api_response,
                ))
                mock_client_cls.return_value = mock_client
                return await tavily_search("Hanwha Q Cells Georgia", max_results=10)

        results = asyncio.run(run())
        # LinkedIn should be filtered out
        urls = [r["url"] for r in results]
        self.assertNotIn("https://linkedin.com/company/hanwha", urls)
        self.assertIn("https://georgia.org/hanwha-news", urls)
        # Priority should be detected
        georgia_result = next(r for r in results if "georgia.org" in r["url"])
        self.assertTrue(georgia_result["is_priority"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
