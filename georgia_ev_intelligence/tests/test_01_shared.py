"""
Step 1 Test: Validate shared/ layer.
Tests: config loading, DB connection, table creation, B2 connection.
Run from georgia_ev_intelligence/ directory.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestConfig(unittest.TestCase):
    """Test that all required config values are loaded correctly."""

    def setUp(self):
        from shared.config import Config
        Config.reset()
        self.cfg = Config.get()

    def test_tavily_key_present(self):
        key = self.cfg.tavily_api_key
        self.assertTrue(key.startswith("tvly-"), f"Expected tvly- prefix, got: {key[:10]}")

    def test_qdrant_url_present(self):
        url = self.cfg.qdrant_url
        self.assertIn("qdrant.io", url, f"Unexpected Qdrant URL: {url}")

    def test_qdrant_dimensions(self):
        self.assertEqual(self.cfg.qdrant_dimensions, 768)

    def test_neo4j_uri_present(self):
        uri = self.cfg.neo4j_uri
        self.assertIn("neo4j", uri.lower(), f"Unexpected Neo4j URI: {uri}")

    def test_neo4j_username_is_neo4j(self):
        # Must be 'neo4j' not the instance ID
        self.assertEqual(self.cfg.neo4j_username, "neo4j")

    def test_database_url_is_postgres(self):
        url = self.cfg.database_url
        self.assertTrue(
            url.startswith("postgresql://"),
            f"Expected postgresql:// prefix, got: {url[:30]}"
        )

    def test_ollama_models_set(self):
        # qwen3:14b not yet pulled — qwen2.5:14b is the installed equivalent
        model = self.cfg.ollama_llm_model.lower()
        self.assertTrue(
            "qwen" in model or "llama" in model,
            f"Expected a qwen or llama model, got: {model}"
        )
        self.assertIn("nomic", self.cfg.ollama_embed_model.lower())

    def test_b2_bucket_set(self):
        bucket = self.cfg.b2_bucket
        self.assertTrue(len(bucket) > 0, "B2 bucket name is empty")


class TestDatabaseConnection(unittest.TestCase):
    """Test PostgreSQL connectivity and table creation."""

    def test_connection_is_alive(self):
        from shared.db import verify_connection
        ok = verify_connection()
        self.assertTrue(ok, "PostgreSQL connection failed — check DATABASE_URL in .env")

    def test_tables_can_be_created(self):
        from shared.db import create_tables
        # Should not raise
        create_tables()

    def test_company_table_schema(self):
        from shared.db import create_tables, get_engine
        from sqlalchemy import inspect as sa_inspect
        create_tables()  # Ensure tables exist first
        engine = get_engine()
        inspector = sa_inspect(engine)
        # Check in all schemas (Supabase uses 'public')
        tables = inspector.get_table_names(schema='public')
        self.assertIn("gev_companies", tables, f"gev_companies not found. Tables: {tables}")

    def test_document_table_schema(self):
        from shared.db import create_tables, get_engine
        from sqlalchemy import inspect as sa_inspect
        create_tables()
        engine = get_engine()
        inspector = sa_inspect(engine)
        tables = inspector.get_table_names(schema='public')
        self.assertIn("gev_documents", tables, f"gev_documents not found. Tables: {tables}")

    def test_extracted_facts_table_schema(self):
        from shared.db import create_tables, get_engine
        from sqlalchemy import inspect as sa_inspect
        create_tables()
        engine = get_engine()
        inspector = sa_inspect(engine)
        tables = inspector.get_table_names(schema='public')
        self.assertIn("gev_extracted_facts", tables, f"gev_extracted_facts not found. Tables: {tables}")

    def test_session_can_be_created_and_closed(self):
        from shared.db import get_session
        session = get_session()
        session.close()  # Should not raise


class TestB2Connection(unittest.TestCase):
    """Test Backblaze B2 connectivity."""

    def test_connection_is_alive(self):
        from shared.storage import verify_connection
        ok = verify_connection()
        self.assertTrue(ok, "Backblaze B2 connection failed — check B2_* keys in .env")

    def test_key_naming_convention(self):
        from shared.storage import make_document_key
        key = make_document_key("Hanwha Q Cells", "abc123def456", ".pdf")
        self.assertTrue(key.startswith("documents/hanwha_q_cells/"))
        self.assertTrue(key.endswith(".pdf"))
        self.assertIn("abc123def456", key)

    def test_slugify_special_chars(self):
        from shared.storage import make_document_key
        key = make_document_key("SK On (Battery) Ltd.", "hash123", ".html")
        # Should not contain special chars
        self.assertNotIn("(", key)
        self.assertNotIn(")", key)
        self.assertNotIn(" ", key)


class TestOllamaConnection(unittest.TestCase):
    """Test that Ollama is running and models are available."""

    def test_ollama_is_reachable(self):
        import httpx
        from shared.config import Config
        cfg = Config.get()
        try:
            response = httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=5.0)
            self.assertEqual(response.status_code, 200, "Ollama returned non-200")
        except httpx.ConnectError:
            self.fail(f"Cannot reach Ollama at {cfg.ollama_base_url} — is Ollama running?")

    def test_embed_model_available(self):
        import httpx
        from shared.config import Config
        cfg = Config.get()
        response = httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=5.0)
        models = [m["name"] for m in response.json().get("models", [])]
        model_names_lower = [m.lower() for m in models]
        self.assertTrue(
            any("nomic" in m for m in model_names_lower),
            f"nomic-embed-text not found in Ollama models: {models}"
        )

    def test_llm_model_available(self):
        import httpx
        from shared.config import Config
        cfg = Config.get()
        response = httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=5.0)
        models = [m["name"] for m in response.json().get("models", [])]
        model_names_lower = [m.lower() for m in models]
        llm_model = cfg.ollama_llm_model.lower().split(":")[0]  # e.g. "qwen3"
        self.assertTrue(
            any(llm_model in m for m in model_names_lower),
            f"{cfg.ollama_llm_model} not found in Ollama models: {models}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
