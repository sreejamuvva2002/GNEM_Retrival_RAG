import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "georgia_ev_intelligence"
KB_DIR = ROOT / "kb"

GNEM_EXCEL = KB_DIR / "GNEM - Auto Landscape Lat Long Updated.xlsx"
HUMAN_QA_EXCEL = KB_DIR / "Human validated 50 questions.xlsx"
OUTPUTS_DIR = PACKAGE_DIR / "outputs"
SMOKE_TEST_OUTPUTS_DIR = OUTPUTS_DIR / "smoke_test"

load_dotenv(ROOT / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:32b")

USE_ANTHROPIC = (
    os.getenv("USE_ANTHROPIC", "false").lower() == "true"
    and bool(ANTHROPIC_API_KEY)
)

SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.65"))
MAX_EVIDENCE_ROWS = int(os.getenv("MAX_EVIDENCE_ROWS", "50"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_LOCAL_FILES_ONLY = os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "false").lower() == "true"
EMBEDDING_TRUST_REMOTE_CODE = os.getenv("EMBEDDING_TRUST_REMOTE_CODE", "false").lower() == "true"
EMBEDDING_DOCUMENT_PREFIX = os.getenv(
    "EMBEDDING_DOCUMENT_PREFIX",
    "search_document: " if "nomic" in EMBEDDING_MODEL.lower() else "",
)
EMBEDDING_QUERY_PREFIX = os.getenv(
    "EMBEDDING_QUERY_PREFIX",
    "search_query: " if "nomic" in EMBEDDING_MODEL.lower() else "",
)
RAG_TOP_K       = int(os.getenv("RAG_TOP_K", "15"))

# Qdrant parent-child chunk index
QDRANT_URL = os.getenv("QDRANT_URL", "https://c4d6dfe9-77b3-48b7-bdd3-cb6dbaf9cdb8.us-east4-0.gcp.cloud.qdrant.io")
QDRANT_PATH = os.getenv("QDRANT_PATH", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MTk5NjRkNWYtMTE2NC00ODFmLWI4YWItOWIwNmY2NTVkOTQxIn0.2RxGNGkawXV-hVocpKSKYKi1T7HOv0DjYFR02NFhEiI")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "georgia_ev_kb_chunks")
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "30"))
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "64"))
USE_QDRANT_RETRIEVER = os.getenv("USE_QDRANT_RETRIEVER", "true").lower() == "true"

# PostgreSQL parent chunk storage
DATABASE_URL = os.getenv("DATABASE_URL", "")

QUERY_REWRITER_MODEL   = os.getenv("QUERY_REWRITER_MODEL", "qwen2.5:32b")
QUERY_REWRITER_TIMEOUT = int(os.getenv("QUERY_REWRITER_TIMEOUT", "60"))
QUERY_REWRITER_ENABLED = os.getenv("QUERY_REWRITER_ENABLED", "true").lower() == "true"
MAX_REWRITER_RETRIES   = int(os.getenv("MAX_REWRITER_RETRIES", "2"))

# Probe retrieval (Stage 1 multi-probe, high-recall)
PROBE_TOP_K_SEMANTIC = int(os.getenv("PROBE_TOP_K_SEMANTIC", os.getenv("PROBE_TOP_K_DENSE", "50")))
PROBE_TOP_K_BM25     = int(os.getenv("PROBE_TOP_K_BM25", "50"))
PROBE_TOP_K_COLUMN   = int(os.getenv("PROBE_TOP_K_COLUMN", "50"))
PROBE_FUSED_TOP_K    = int(os.getenv("PROBE_FUSED_TOP_K", "150"))
PROBE_MIN_ROWS       = int(os.getenv("PROBE_MIN_ROWS", "10"))

# KB term extraction
KB_TERM_MIN_FREQUENCY  = int(os.getenv("KB_TERM_MIN_FREQUENCY", "2"))
KB_TERM_TOP_N          = int(os.getenv("KB_TERM_TOP_N", "30"))
KB_TERM_MIN_DISCOVERED = int(os.getenv("KB_TERM_MIN_DISCOVERED", "3"))
