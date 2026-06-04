"""Central configuration: loads all environment variables from .env.

WHY THIS FILE EXISTS:
  Single source of truth for every runtime setting.  All modules import
  this file (as `from georgia_ev_intelligence.shared import config`) rather
  than calling os.environ directly, so settings can be changed in one place.

ENVIRONMENT VARIABLES (defined in .env — see .env.example):
  NEON_DATABASE_URL       — PostgreSQL connection string (Neon cloud DB)
                            Used by BM25 retriever, dense retriever, parent fetcher
  OLLAMA_BASE_URL         — URL of the local Ollama server (default: localhost:11434)
  OLLAMA_LLM_MODEL        — Default model for llm_client.py (not used by run_baseline)
  OLLAMA_TEMPERATURE      — Sampling temperature (default: 0.1 — nearly deterministic)
  OLLAMA_TOP_P            — Nucleus sampling threshold (default: 0.9)
  OLLAMA_NUM_PREDICT      — Max tokens to generate (default: 4096)
  EMBEDDING_MODEL         — HuggingFace model ID for child-chunk embeddings
                            (nomic-ai/nomic-embed-text-v1.5)
  EMBEDDING_LOCAL_FILES_ONLY — If true, never downloads from HuggingFace
  EMBEDDING_TRUST_REMOTE_CODE — Required true for nomic-embed-text
  EMBEDDING_DOCUMENT_PREFIX   — Prefix for indexing ("search_document:")
  EMBEDDING_QUERY_PREFIX      — Prefix for queries ("search_query:")
  PGVECTOR_BATCH_SIZE     — Batch size for pgvector upsert operations

PATH CONSTANTS:
  GNEM_EXCEL      — outputs/Normalized_kb.xlsx (used by direct_kb_pipeline)
  HUMAN_QA_EXCEL  — kb/Human validated 50 questions.xlsx

RELATIONSHIPS:
  Imported by virtually every runtime and offline module.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = ROOT / "georgia_ev_intelligence"
KB_DIR = ROOT / "kb"
OUTPUTS_DIR = PACKAGE_DIR / "outputs"
RAW_DOCS_DIR = KB_DIR / "raw_docs"

GNEM_EXCEL = OUTPUTS_DIR / "Normalized_kb.xlsx"
HUMAN_QA_EXCEL = KB_DIR / "Human validated 50 questions.xlsx"
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


def _env_optional_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_optional_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# Neon PostgreSQL (parent chunks storage)
NEON_DATABASE_URL = _env("NEON_DATABASE_URL")

OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL")
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:14b")
OLLAMA_TEMPERATURE = _env_optional_float("OLLAMA_TEMPERATURE", 0.1)
OLLAMA_TOP_P = _env_optional_float("OLLAMA_TOP_P", 0.9)
OLLAMA_NUM_PREDICT = _env_optional_int("OLLAMA_NUM_PREDICT", 4096)

EMBEDDING_MODEL = _env("EMBEDDING_MODEL")
EMBEDDING_LOCAL_FILES_ONLY = _env_bool("EMBEDDING_LOCAL_FILES_ONLY")
EMBEDDING_TRUST_REMOTE_CODE = _env_bool("EMBEDDING_TRUST_REMOTE_CODE")
EMBEDDING_DOCUMENT_PREFIX = _env("EMBEDDING_DOCUMENT_PREFIX")
EMBEDDING_QUERY_PREFIX = _env("EMBEDDING_QUERY_PREFIX")

# pgvector child chunk index (Neon PostgreSQL)
PGVECTOR_BATCH_SIZE = _env_int("PGVECTOR_BATCH_SIZE")

# ---------------------------------------------------------------------------
# kb_builder crawler settings (all optional — safe defaults provided)
# ---------------------------------------------------------------------------

def _env_optional_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


CRAWLER_CONCURRENCY: int = _env_optional_int("CRAWLER_CONCURRENCY", 5)
CRAWLER_DEPTH: int = _env_optional_int("CRAWLER_DEPTH", 3)
CRAWLER_DELAY_SECONDS: float = _env_optional_float("CRAWLER_DELAY_SECONDS", 1.0)
CRAWLER_USER_AGENT: str = _env_optional_str(
    "CRAWLER_USER_AGENT", "GNEM-RAG-Bot/1.0 (research crawler)"
)
# Cron expression for the periodic re-crawl scheduler (default: every Sunday at 02:00)
CRAWLER_SCHEDULE_CRON: str = _env_optional_str("CRAWLER_SCHEDULE_CRON", "0 2 * * 0")

# ---------------------------------------------------------------------------
# Backblaze B2 storage (optional — leave blank to disable B2 upload)
# ---------------------------------------------------------------------------
B2_KEY_ID:          str = _env_optional_str("B2_KEY_ID", "")
B2_APPLICATION_KEY: str = _env_optional_str("B2_APPLICATION_KEY", "")
B2_BUCKET_NAME:     str = _env_optional_str("B2_BUCKET_NAME", "")
B2_ENDPOINT_URL:    str = _env_optional_str("B2_ENDPOINT_URL", "")

# ---------------------------------------------------------------------------
# markdown_extraction pipeline (raw B2 documents → Markdown corpus)
# All optional — safe defaults provided.
# ---------------------------------------------------------------------------

def _env_optional_list(name: str, default: str) -> list[str]:
    """Comma-separated env var → list of stripped, non-empty strings."""
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# B2 prefixes the crawler writes raw bytes to (flat, one per file type).
RAW_B2_PREFIXES: list[str] = _env_optional_list(
    "RAW_B2_PREFIXES",
    "raw-html,raw-pdf,raw-docx,raw-excel,raw-csv,raw-json,raw-xml,raw-text,raw-image",
)
# Where converted Markdown + manifests live (never overlap the raw prefixes).
MARKDOWN_B2_PREFIX: str = _env_optional_str("MARKDOWN_B2_PREFIX", "processed/markdown/v1/")
MANIFEST_B2_PREFIX: str = _env_optional_str("MANIFEST_B2_PREFIX", "manifests/")

# Local mirrors (authoritative storage remains B2).
LOCAL_RAW_CACHE: Path = ROOT / _env_optional_str("LOCAL_RAW_CACHE", "data/raw_cache/web_documents")
LOCAL_MARKDOWN_DIR: Path = ROOT / _env_optional_str("LOCAL_MARKDOWN_DIR", "data/processed/markdown/v1")
LOCAL_MANIFEST_DIR: Path = ROOT / _env_optional_str("LOCAL_MANIFEST_DIR", "data/manifests")

EXTRACTION_VERSION: str = _env_optional_str("EXTRACTION_VERSION", "v1")
MAX_FILE_SIZE_MB: int = _env_optional_int("MAX_FILE_SIZE_MB", 100)
