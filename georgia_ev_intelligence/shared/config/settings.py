import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = ROOT / "georgia_ev_intelligence"
KB_DIR = ROOT / "kb"

GNEM_EXCEL = KB_DIR / "GNEM - Auto Landscape Lat Long Updated.xlsx"
HUMAN_QA_EXCEL = KB_DIR / "Human validated 50 questions.xlsx"
OUTPUTS_DIR = PACKAGE_DIR / "outputs"
SMOKE_TEST_OUTPUTS_DIR = OUTPUTS_DIR / "smoke_test"

load_dotenv(ROOT / ".env")


def _env(name: str) -> str:
    if name not in os.environ:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return os.environ[name]


def _env_bool(name: str) -> bool:
    return _env(name).lower() == "true"


def _env_int(name: str) -> int:
    return int(_env(name))


def _env_float(name: str) -> float:
    return float(_env(name))


# Neon PostgreSQL (parent chunks storage)
NEON_DATABASE_URL = _env("NEON_DATABASE_URL")

ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = _env("ANTHROPIC_MODEL")

OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL")
OLLAMA_LLM_MODEL = _env("OLLAMA_LLM_MODEL")

USE_ANTHROPIC = (
    _env_bool("USE_ANTHROPIC")
    and bool(ANTHROPIC_API_KEY)
)

SEMANTIC_THRESHOLD = _env_float("SEMANTIC_THRESHOLD")
MAX_EVIDENCE_ROWS = _env_int("MAX_EVIDENCE_ROWS")

EMBEDDING_MODEL = _env("EMBEDDING_MODEL")
EMBEDDING_LOCAL_FILES_ONLY = _env_bool("EMBEDDING_LOCAL_FILES_ONLY")
EMBEDDING_TRUST_REMOTE_CODE = _env_bool("EMBEDDING_TRUST_REMOTE_CODE")
EMBEDDING_DOCUMENT_PREFIX = _env("EMBEDDING_DOCUMENT_PREFIX")
EMBEDDING_QUERY_PREFIX = _env("EMBEDDING_QUERY_PREFIX")
RAG_TOP_K = _env_int("RAG_TOP_K")

# pgvector child chunk index (Neon PostgreSQL)
PGVECTOR_BATCH_SIZE = _env_int("PGVECTOR_BATCH_SIZE")
USE_PGVECTOR_RETRIEVER = _env_bool("USE_PGVECTOR_RETRIEVER")

QUERY_REWRITER_MODEL = _env("QUERY_REWRITER_MODEL")
QUERY_REWRITER_TIMEOUT = _env_int("QUERY_REWRITER_TIMEOUT")
QUERY_REWRITER_ENABLED = _env_bool("QUERY_REWRITER_ENABLED")
MAX_REWRITER_RETRIES = _env_int("MAX_REWRITER_RETRIES")

# Probe retrieval (Stage 1 multi-probe, high-recall)
PROBE_TOP_K_SEMANTIC = _env_int("PROBE_TOP_K_SEMANTIC")
PROBE_TOP_K_BM25 = _env_int("PROBE_TOP_K_BM25")
PROBE_TOP_K_COLUMN = _env_int("PROBE_TOP_K_COLUMN")
PROBE_FUSED_TOP_K = _env_int("PROBE_FUSED_TOP_K")
PROBE_MIN_ROWS = _env_int("PROBE_MIN_ROWS")

# KB term extraction
KB_TERM_MIN_FREQUENCY = _env_int("KB_TERM_MIN_FREQUENCY")
KB_TERM_TOP_N = _env_int("KB_TERM_TOP_N")
KB_TERM_MIN_DISCOVERED = _env_int("KB_TERM_MIN_DISCOVERED")
