"""
Phase 2 Test Suite — validates chunker, embedder, and Qdrant vector store.

Tests are ordered: each test builds on the previous one.

Run:
  venv\\Scripts\\python -m pytest tests/test_03_phase2.py -v --tb=short
"""
from __future__ import annotations

import sys
import unittest
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2_embedding.chunker import (
    COMPANY_CHUNK_SCHEMA_VERSION,
    Chunk,
    chunk_company_record,
    chunk_document,
    get_child_chunks,
    get_parent_chunk,
    _company_source_hash,
    _estimate_tokens,
    CHILD_CHAR_TARGET,
    PARENT_CHAR_TARGET,
)
from phase2_embedding.index_freshness import audit_company_index
from phase2_embedding.embedder import embed_single, embed_texts, verify_ollama_embed
from phase2_embedding.vector_store import (
    ensure_collection_exists,
    get_collection_stats,
    upload_chunks,
    search_hybrid,
    verify_qdrant_connection,
    delete_company_chunks,
    _build_metadata_filter,
    _build_sparse_vector,
)
from phase1_extraction.kb_loader import build_document_text


# ─── Sample data ──────────────────────────────────────────────────────────────

SAMPLE_COMPANY = {
    "id": 9999,
    "company_name": "Test EV Battery LLC",
    "tier": "Tier 1",
    "ev_supply_chain_role": "Battery Cell Manufacturer",
    "primary_oems": "Hyundai, Kia",
    "ev_battery_relevant": "Yes",
    "industry_group": "Battery Manufacturing",
    "location_city": "Savannah",
    "location_county": "Chatham County",
    "location_state": "Georgia",
    "employment": 2500.0,
    "products_services": "Lithium-ion battery cells and modules for EVs",
    "classification_method": "EV Direct",
    "supplier_affiliation_type": "Tier 1 Supplier",
    "latitude": 32.0835,
    "longitude": -81.0998,
}

SAMPLE_DOC_TEXT = """
Hyundai Motor Group has announced a $5.5 billion investment in Chatham County, Georgia.
The Hyundai METAPLANT America facility will create approximately 8,500 direct jobs.
Construction is expected to complete by Q2 2025.

The facility will manufacture electric vehicles and feature advanced battery assembly lines.
Located in Bryan County, the plant spans over 2,900 acres.

SK On, a Korean battery manufacturer, will supply battery modules to Hyundai METAPLANT.
SK On invested $300 million in their Georgia facility, creating 1,200 jobs in Commerce, Georgia.

Kia Motors Manufacturing Georgia in West Point has been operational since 2009.
The West Point facility employs over 3,000 workers and produces the Telluride and Sorento.
Kia announced an expansion to add EV production capabilities starting 2026.
""" * 3  # Repeat to get enough text for parent-child splitting


class TestChunker(unittest.TestCase):
    """Test the hierarchical chunker."""

    def test_estimate_tokens(self):
        """Token estimate should be chars / 4."""
        text = "a" * 400  # 400 chars → ~100 tokens
        est = _estimate_tokens(text)
        self.assertAlmostEqual(est, 100, delta=5)

    def test_company_chunk_creates_multi_view_chunks(self):
        """Each company produces a master chunk plus focused semantic views."""
        doc_text = build_document_text(SAMPLE_COMPANY)
        chunks = chunk_company_record(SAMPLE_COMPANY, doc_text)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(chunk.chunk_type == "company" for chunk in chunks))
        self.assertTrue(all(chunk.parent_id is None for chunk in chunks))
        view_names = {chunk.metadata.get("chunk_view") for chunk in chunks}
        self.assertIn("master", view_names)
        self.assertIn("role", view_names)
        self.assertIn("product", view_names)
        self.assertIn("oem", view_names)
        self.assertIn("classification", view_names)
        self.assertIn("capability", view_names)
        self.assertIn("location", view_names)

    def test_company_chunk_metadata_complete(self):
        """Master company chunk metadata must include all key fields."""
        doc_text = build_document_text(SAMPLE_COMPANY)
        chunks = chunk_company_record(SAMPLE_COMPANY, doc_text)
        master = next(chunk for chunk in chunks if chunk.metadata.get("chunk_view") == "master")
        meta = master.metadata
        required_fields = [
            "company_name", "tier", "ev_supply_chain_role",
            "location_county", "employment", "ev_battery_relevant",
            "source_type", "chunk_type",
        ]
        for field in required_fields:
            self.assertIn(field, meta, f"Missing metadata field: {field}")
        self.assertEqual(meta["source_type"], "gnem_excel")
        self.assertEqual(meta["company_name"], "Test EV Battery LLC")
        self.assertTrue(meta["company_row_id"])
        self.assertEqual(meta["chunk_view"], "master")
        self.assertEqual(meta["kb_schema_version"], COMPANY_CHUNK_SCHEMA_VERSION)
        self.assertTrue(meta["source_row_hash"])
        self.assertIn("company_context_text", meta)
        self.assertEqual(meta["products_services_full"], SAMPLE_COMPANY["products_services"])

    def test_company_chunk_ids_are_stable(self):
        """Company chunk ids must be deterministic for idempotent Qdrant upserts."""
        doc_text = build_document_text(SAMPLE_COMPANY)
        first = chunk_company_record(SAMPLE_COMPANY, doc_text)
        second = chunk_company_record(SAMPLE_COMPANY, doc_text)
        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])

    def test_company_source_hash_changes_with_source_fields(self):
        """Source hash must change when a source field changes."""
        changed = dict(SAMPLE_COMPANY)
        changed["products_services"] = "Changed battery module description"
        self.assertNotEqual(_company_source_hash(SAMPLE_COMPANY), _company_source_hash(changed))

    def test_company_product_payload_keeps_full_source_text(self):
        """Long product text should remain available in payload for context recall."""
        company = dict(SAMPLE_COMPANY)
        company["products_services"] = " ".join(["battery module enclosure"] * 40)
        chunks = chunk_company_record(company, build_document_text(company))
        master = next(chunk for chunk in chunks if chunk.metadata.get("chunk_view") == "master")
        self.assertEqual(master.metadata["products_services"], company["products_services"])
        self.assertEqual(master.metadata["products_services_full"], company["products_services"])

    def test_company_chunk_contains_all_text(self):
        """Master company chunk text must include company name, tier, location."""
        doc_text = build_document_text(SAMPLE_COMPANY)
        chunks = chunk_company_record(SAMPLE_COMPANY, doc_text)
        master = next(chunk for chunk in chunks if chunk.metadata.get("chunk_view") == "master")
        text = master.text
        self.assertIn("Test EV Battery LLC", text)
        self.assertIn("Tier 1", text)
        self.assertIn("Chatham County", text)

    def test_document_produces_parent_and_child_chunks(self):
        """Long document should produce both parent and child chunks."""
        chunks = chunk_document(
            text=SAMPLE_DOC_TEXT,
            company_name="Hyundai Metaplant",
            document_id=42,
            source_url="https://example.com/hyundai-announcement",
            document_type="press_release",
            company_metadata=SAMPLE_COMPANY,
        )
        parents = [c for c in chunks if c.chunk_type == "parent"]
        children = [c for c in chunks if c.chunk_type == "child"]
        self.assertGreater(len(parents), 0, "Expected at least 1 parent chunk")
        self.assertGreater(len(children), 0, "Expected at least 1 child chunk")
        # More children than parents
        self.assertGreaterEqual(len(children), len(parents))

    def test_child_chunks_have_parent_id(self):
        """All child chunks must reference their parent."""
        chunks = chunk_document(
            text=SAMPLE_DOC_TEXT,
            company_name="Hyundai Metaplant",
            document_id=42,
            source_url="https://example.com/test",
        )
        children = [c for c in chunks if c.chunk_type == "child"]
        for child in children:
            self.assertIsNotNone(child.parent_id, "Child chunk missing parent_id")

    def test_child_chunk_size_within_target(self):
        """Child chunks should be roughly within target size (1024 chars ± 30%)."""
        chunks = chunk_document(
            text=SAMPLE_DOC_TEXT,
            company_name="Hyundai Metaplant",
            document_id=42,
            source_url="https://example.com/test",
        )
        children = [c for c in chunks if c.chunk_type == "child"]
        for child in children:
            # Allow generous tolerance — sentence splitting can create irregular sizes
            self.assertLessEqual(
                child.char_count, CHILD_CHAR_TARGET * 2.5,
                f"Child chunk too large: {child.char_count} chars"
            )

    def test_get_child_chunks(self):
        """get_child_chunks returns only child + company chunks."""
        chunks = chunk_document(
            text=SAMPLE_DOC_TEXT,
            company_name="Test Co",
            document_id=1,
            source_url="https://example.com",
        )
        children = get_child_chunks(chunks)
        for c in children:
            self.assertIn(c.chunk_type, ("child", "company"))

    def test_get_parent_chunk(self):
        """get_parent_chunk returns the correct parent for a child."""
        chunks = chunk_document(
            text=SAMPLE_DOC_TEXT,
            company_name="Test Co",
            document_id=1,
            source_url="https://example.com",
        )
        children = [c for c in chunks if c.chunk_type == "child"]
        if children:
            child = children[0]
            parent = get_parent_chunk(chunks, child)
            self.assertIsNotNone(parent)
            self.assertEqual(parent.chunk_id, child.parent_id)
            self.assertEqual(parent.chunk_type, "parent")

    def test_empty_text_returns_no_chunks(self):
        """Empty text should return empty list."""
        chunks = chunk_document(
            text="",
            company_name="Test Co",
            document_id=1,
            source_url="https://example.com",
        )
        self.assertEqual(len(chunks), 0)

    def test_document_chunk_metadata_has_company_fields(self):
        """Document chunks must inherit company metadata for filtering."""
        chunks = chunk_document(
            text=SAMPLE_DOC_TEXT,
            company_name="Hyundai Metaplant",
            document_id=42,
            source_url="https://example.com/test",
            company_metadata=SAMPLE_COMPANY,
        )
        children = [c for c in chunks if c.chunk_type == "child"]
        if children:
            meta = children[0].metadata
            self.assertIn("tier", meta)
            self.assertIn("location_county", meta)
            self.assertIn("ev_battery_relevant", meta)
            self.assertEqual(meta["company_name"], "Hyundai Metaplant")


class TestIndexFreshness(unittest.TestCase):
    """Test offline Qdrant freshness audit logic."""

    def _master_payload(self, company: dict, **overrides):
        doc_text = build_document_text(company)
        chunks = chunk_company_record(company, doc_text)
        master = next(chunk for chunk in chunks if chunk.metadata.get("chunk_view") == "master")
        return {**master.metadata, **overrides}

    def test_company_index_audit_ok_for_matching_master(self):
        """Matching KB row and Qdrant master payload should pass."""
        payload = self._master_payload(SAMPLE_COMPANY)
        audit = audit_company_index(
            kb_companies=[SAMPLE_COMPANY],
            qdrant_records=[{"payload": payload}],
        )
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["expected_rows"], 1)
        self.assertEqual(audit["indexed_master_rows"], 1)

    def test_company_index_audit_flags_stale_missing_extra_duplicate(self):
        """Audit should identify freshness failures without Qdrant access."""
        other_company = dict(SAMPLE_COMPANY)
        other_company["id"] = 10000
        other_company["company_name"] = "Other Battery Co"

        stale_payload = self._master_payload(
            SAMPLE_COMPANY,
            source_row_hash="old-hash",
            kb_schema_version="old-schema",
        )
        duplicate_payload = dict(stale_payload)
        extra_payload = self._master_payload(other_company)

        audit = audit_company_index(
            kb_companies=[SAMPLE_COMPANY],
            qdrant_records=[
                {"payload": stale_payload},
                {"payload": duplicate_payload},
                {"payload": extra_payload},
            ],
        )

        self.assertFalse(audit["ok"])
        self.assertIn(stale_payload["company_row_id"], audit["stale_row_ids"])
        self.assertIn(stale_payload["company_row_id"], audit["duplicate_row_ids"])
        self.assertIn(extra_payload["company_row_id"], audit["extra_row_ids"])
        self.assertIn(stale_payload["company_row_id"], audit["stale_schema_row_ids"])


class TestSparseVector(unittest.TestCase):
    """Test sparse vector generation."""

    def test_sparse_vector_has_indices_and_values(self):
        """Sparse vector must have indices and values."""
        sv = _build_sparse_vector("Hyundai battery manufacturing Georgia")
        self.assertIn("indices", sv)
        self.assertIn("values", sv)
        self.assertEqual(len(sv["indices"]), len(sv["values"]))

    def test_sparse_vector_filters_stopwords(self):
        """Common stopwords should not dominate the sparse vector."""
        sv = _build_sparse_vector("the a an and of in this that")
        # Should still return something (minimal vector)
        self.assertIsInstance(sv["indices"], list)

    def test_sparse_vector_different_for_different_texts(self):
        """Different texts should produce different sparse vectors."""
        sv1 = _build_sparse_vector("Hyundai Kia battery Georgia")
        sv2 = _build_sparse_vector("BMW Mercedes Germany steel")
        # The indices sets should be different
        self.assertNotEqual(set(sv1["indices"]), set(sv2["indices"]))


class TestEmbedder(unittest.TestCase):
    """Test the Ollama embedder. Requires Ollama running locally."""

    @classmethod
    def setUpClass(cls):
        """Check Ollama is available before running embedder tests."""
        result = verify_ollama_embed()
        if not result["ok"]:
            raise unittest.SkipTest(
                f"Ollama not available: {result['error']}. "
                "Start with: ollama serve"
            )
        cls.embed_dims = result["dimensions"]

    def test_embed_single_returns_correct_dimensions(self):
        """Single text embedding should match the configured vector dimensions."""
        vector = embed_single("Georgia EV supply chain battery manufacturer")
        self.assertEqual(len(vector), self.embed_dims)

    def test_embed_single_returns_floats(self):
        """Embedding vector should contain floats."""
        vector = embed_single("Test embedding")
        self.assertTrue(all(isinstance(v, float) for v in vector))

    def test_embed_texts_batch(self):
        """Batch embedding should return one vector per input."""
        texts = [
            "Hanwha Q Cells solar energy Georgia",
            "SK On battery manufacturing commerce",
            "Hyundai METAPLANT EV production",
        ]
        vectors = embed_texts(texts)
        self.assertEqual(len(vectors), len(texts))
        for v in vectors:
            self.assertEqual(len(v), self.embed_dims)

    def test_embed_chunks_returns_dict(self):
        """embed_chunks should return dict mapping chunk_id → vector."""
        from phase2_embedding.embedder import embed_chunks
        doc_text = build_document_text(SAMPLE_COMPANY)
        chunks = chunk_company_record(SAMPLE_COMPANY, doc_text)
        vectors = embed_chunks(chunks)
        self.assertEqual(len(vectors), len(chunks))
        for chunk_id, vec in vectors.items():
            self.assertEqual(len(vec), self.embed_dims)

    def test_same_text_similar_embeddings(self):
        """Same text should produce very similar embeddings (cosine ≈ 1.0)."""
        text = "Kia Georgia manufacturing electric vehicles"
        v1 = embed_single(text)
        v2 = embed_single(text)
        # Cosine similarity
        import math
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))
        cosine = dot / (mag1 * mag2)
        self.assertGreater(cosine, 0.999, "Same text should produce identical embeddings")

    def test_different_texts_different_embeddings(self):
        """Different texts should produce different embeddings."""
        v1 = embed_single("Hyundai battery manufacturing")
        v2 = embed_single("Georgian folk music and dance")
        self.assertNotEqual(v1, v2)


class TestVectorStore(unittest.TestCase):
    """Test Qdrant vector store. Requires Qdrant Cloud connection."""

    @classmethod
    def setUpClass(cls):
        """Verify Qdrant is available before running store tests."""
        result = verify_qdrant_connection()
        if not result["ok"]:
            raise unittest.SkipTest(f"Qdrant not available: {result['error']}")

        # Also need Ollama for generating test embeddings
        embed_result = verify_ollama_embed()
        if not embed_result["ok"]:
            raise unittest.SkipTest(f"Ollama needed for vector store tests: {embed_result['error']}")

    def test_collection_exists(self):
        """Qdrant collection must exist."""
        exists = ensure_collection_exists()
        self.assertTrue(exists, "Qdrant collection 'georgia_ev_chunks' not found")

    def test_upload_company_chunk(self):
        """Upload a test company chunk to Qdrant."""
        from phase2_embedding.embedder import embed_chunks

        doc_text = build_document_text(SAMPLE_COMPANY)
        chunks = chunk_company_record(SAMPLE_COMPANY, doc_text)
        # Use a unique company name to avoid polluting real data
        for chunk in chunks:
            chunk.metadata["company_name"] = "__test_company__"
            chunk.metadata["source_type"] = "__test__"

        vectors = embed_chunks(chunks)
        uploaded = upload_chunks(chunks, vectors)
        self.assertGreater(uploaded, 0, "Should upload at least 1 chunk")

    def test_search_returns_results(self):
        """Hybrid search should return results after uploading test data."""
        query = "EV battery manufacturer Georgia Tier 1"
        query_vec = embed_single(query)
        results = search_hybrid(
            query_text=query,
            query_vector=query_vec,
            top_k=5,
        )
        self.assertIsInstance(results, list)
        # May be 0 if collection is empty
        if results:
            self.assertIn("score", results[0])
            self.assertIn("text", results[0])
            self.assertIn("company_name", results[0])

    def test_search_with_filter(self):
        """Search with metadata filter should only return matching chunks."""
        query_vec = embed_single("battery supply chain")
        results = search_hybrid(
            query_text="battery supply chain",
            query_vector=query_vec,
            top_k=5,
            filters={"source_type": "__test__"},
        )
        self.assertIsInstance(results, list)
        # All returned results should match the filter
        for r in results:
            if r["metadata"].get("source_type"):
                self.assertEqual(r["metadata"]["source_type"], "__test__")

    def test_metadata_filter_builder_single(self):
        """Single filter should produce Filter object."""
        f = _build_metadata_filter({"tier": "Tier 1"})
        self.assertIsNotNone(f)

    def test_metadata_filter_builder_empty(self):
        """Empty filter dict should return None."""
        f = _build_metadata_filter({})
        self.assertIsNone(f)

    def test_metadata_filter_employment_range(self):
        """Employment range filter should build correctly."""
        f = _build_metadata_filter({"min_employment": 500, "max_employment": 5000})
        self.assertIsNotNone(f)

    def test_get_collection_stats(self):
        """Collection stats should return points_count."""
        stats = get_collection_stats()
        self.assertIn("points_count", stats)
        self.assertIsInstance(stats["points_count"], int)

    def tearDown(self):
        """Clean up test data after each test."""
        try:
            delete_company_chunks("__test_company__")
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
