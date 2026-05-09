import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
KB_DIR = ROOT / "kb"

GNEM_EXCEL = KB_DIR / "GNEM - Auto Landscape Lat Long Updated.xlsx"
HUMAN_QA_EXCEL = KB_DIR / "Human validated 50 questions.xlsx"
EMPLOYMENT_OVERRIDES = KB_DIR / "employment_overrides.csv"
OUTPUTS_DIR = Path(__file__).parent / "outputs"

load_dotenv(Path(__file__).parent / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:14b")

USE_ANTHROPIC = bool(ANTHROPIC_API_KEY)

SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.65"))
MAX_EVIDENCE_ROWS = int(os.getenv("MAX_EVIDENCE_ROWS", "50"))

EMBEDDING_MODEL        = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
REACT_MAX_ITERATIONS   = int(os.getenv("REACT_MAX_ITERATIONS", "5"))
REACT_TOP_K            = int(os.getenv("REACT_TOP_K", "15"))
REACT_OLLAMA_TIMEOUT   = int(os.getenv("REACT_OLLAMA_TIMEOUT", "60"))

QUERY_REWRITER_MODEL   = os.getenv("QUERY_REWRITER_MODEL", "qwen2.5:14b")
QUERY_REWRITER_TIMEOUT = int(os.getenv("QUERY_REWRITER_TIMEOUT", "60"))
QUERY_REWRITER_ENABLED = os.getenv("QUERY_REWRITER_ENABLED", "true").lower() == "true"
