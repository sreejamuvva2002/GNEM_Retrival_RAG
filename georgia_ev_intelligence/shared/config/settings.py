import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = ROOT / "georgia_ev_intelligence"
KB_DIR = ROOT / "kb"
OUTPUTS_DIR = PACKAGE_DIR / "outputs"

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
