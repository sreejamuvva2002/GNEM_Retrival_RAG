import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = ROOT / "georgia_ev_intelligence"
KB_DIR = ROOT / "kb"

GNEM_EXCEL = KB_DIR / "GNEM - Auto Landscape Lat Long Updated.xlsx"
HUMAN_QA_EXCEL = KB_DIR / "Human validated 50 questions.xlsx"
EMPLOYMENT_OVERRIDES = KB_DIR / "employment_overrides.csv"
OUTPUTS_DIR = PACKAGE_DIR / "outputs"
SMOKE_TEST_OUTPUTS_DIR = OUTPUTS_DIR / "smoke_test"

load_dotenv(ROOT / ".env")

# Neon PostgreSQL (parent chunks storage)
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL", "")

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

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBEDDING_LOCAL_FILES_ONLY = os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "false").lower() == "true"
EMBEDDING_TRUST_REMOTE_CODE = os.getenv("EMBEDDING_TRUST_REMOTE_CODE", "true").lower() == "true"
EMBEDDING_DOCUMENT_PREFIX = os.getenv(
    "EMBEDDING_DOCUMENT_PREFIX",
    "search_document: " if "nomic" in EMBEDDING_MODEL.lower() else "",
)
EMBEDDING_QUERY_PREFIX = os.getenv(
    "EMBEDDING_QUERY_PREFIX",
    "search_query: " if "nomic" in EMBEDDING_MODEL.lower() else "",
)
RAG_TOP_K       = int(os.getenv("RAG_TOP_K", "15"))

# pgvector child chunk index (Neon PostgreSQL)
PGVECTOR_BATCH_SIZE = int(os.getenv("PGVECTOR_BATCH_SIZE", "64"))
USE_PGVECTOR_RETRIEVER = os.getenv("USE_PGVECTOR_RETRIEVER", "true").lower() == "true"

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
