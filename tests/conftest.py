"""Root pytest configuration.

Sets stub environment variables so settings.py can be imported during test
collection without a real .env file.  All DB and Ollama calls in tests are
mocked — these values are never used for actual connections.
"""
import os

os.environ.setdefault("NEON_DATABASE_URL", "postgresql://test:test@localhost/test")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_LLM_MODEL", "test-model")
os.environ.setdefault("OLLAMA_TEMPERATURE", "0.1")
os.environ.setdefault("OLLAMA_TOP_P", "0.9")
os.environ.setdefault("OLLAMA_NUM_PREDICT", "512")
os.environ.setdefault("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
os.environ.setdefault("EMBEDDING_LOCAL_FILES_ONLY", "false")
os.environ.setdefault("EMBEDDING_TRUST_REMOTE_CODE", "true")
os.environ.setdefault("EMBEDDING_DOCUMENT_PREFIX", "search_document:")
os.environ.setdefault("EMBEDDING_QUERY_PREFIX", "search_query:")
os.environ.setdefault("PGVECTOR_BATCH_SIZE", "64")
