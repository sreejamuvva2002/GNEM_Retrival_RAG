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


def _env_optional_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
# Self-healing RAG loop (closed-loop retrieve → judge → generate → verify →
# bounded retry). All optional — safe defaults make the loop on-by-default.
# Set SELF_HEALING_ENABLED=false to fully restore the open-loop fast path.
# ---------------------------------------------------------------------------
SELF_HEALING_ENABLED: bool = _env_optional_bool("SELF_HEALING_ENABLED", True)
# Total attempts = 1 initial + retries. 3 == 2 retries.
SELF_HEALING_MAX_ATTEMPTS: int = _env_optional_int("SELF_HEALING_MAX_ATTEMPTS", 3)
# How much to grow reranker_top_k each time retrieval is judged insufficient.
SELF_HEALING_WIDEN_STEP: int = _env_optional_int("SELF_HEALING_WIDEN_STEP", 20)
# Decompose multi-part questions into sub-queries before retrieval.
SELF_HEALING_DECOMPOSE_ENABLED: bool = _env_optional_bool(
    "SELF_HEALING_DECOMPOSE_ENABLED", True
)
SELF_HEALING_MAX_SUBQUERIES: int = _env_optional_int("SELF_HEALING_MAX_SUBQUERIES", 4)
# Post-generation LLM groundedness verification (Gate B). Deterministic checks
# always run regardless of this flag.
SELF_HEALING_VERIFY_ENABLED: bool = _env_optional_bool("SELF_HEALING_VERIFY_ENABLED", True)
# How many top parent snippets the judge / verifier see.
SELF_HEALING_JUDGE_SNIPPET_COUNT: int = _env_optional_int(
    "SELF_HEALING_JUDGE_SNIPPET_COUNT", 10
)
# Max chars per snippet shown to the judge / verifier.
SELF_HEALING_SNIPPET_CHARS: int = _env_optional_int("SELF_HEALING_SNIPPET_CHARS", 500)
# Regenerations (no re-retrieve) allowed per attempt before escalating to widen.
SELF_HEALING_REGEN_MAX: int = _env_optional_int("SELF_HEALING_REGEN_MAX", 1)

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

# ---------------------------------------------------------------------------
# Per-question debug tracing → cumulative XLSX (outputs/debug_traces/).
# For every question asked through the UI, write a very detailed, one-row-per-
# step record of what happened at each pipeline step. On by default; set
# DEBUG_TRACE_ENABLED=false to disable. Only the UI dispatch path opens a
# session, so tests / batch scripts never write a trace file.
# ---------------------------------------------------------------------------
DEBUG_TRACE_ENABLED: bool = _env_optional_bool("DEBUG_TRACE_ENABLED", True)
DEBUG_TRACE_PATH: str = _env_optional_str(
    "DEBUG_TRACE_PATH",
    str(OUTPUTS_DIR / "debug_traces" / "ui_question_debug.xlsx"),
)
# Excel caps a cell at 32,767 chars; truncate slightly under that with a marker.
DEBUG_TRACE_MAX_CELL_CHARS: int = _env_optional_int("DEBUG_TRACE_MAX_CELL_CHARS", 32000)
