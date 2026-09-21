"""Runtime settings, loaded from the environment (.env is honoured)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional, only needed for local .env files
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a convenience, not a requirement
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]

MODE_LLM = "llm_only"
MODE_KB = "llm_kb"
MODE_KB_WEB = "llm_kb_web"
MODES = (MODE_LLM, MODE_KB, MODE_KB_WEB)

MODE_LABELS = {
    MODE_LLM: "Gemini only (parametric knowledge)",
    MODE_KB: "Gemini + internal company data",
    MODE_KB_WEB: "Gemini + internal company data + web search",
}


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _path(name: str, default: str) -> Path:
    raw = Path(_env(name, default))
    return raw if raw.is_absolute() else REPO_ROOT / raw


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash"))
    judge_model: str = field(default_factory=lambda: _env("GEMINI_JUDGE_MODEL", "gemini-2.5-pro"))
    embed_model: str = field(default_factory=lambda: _env("GEMINI_EMBED_MODEL", "gemini-embedding-001"))

    temperature: float = field(default_factory=lambda: float(_env("GEMINI_TEMPERATURE", "0.0")))
    max_output_tokens: int = field(default_factory=lambda: int(_env("GEMINI_MAX_OUTPUT_TOKENS", "2048")))
    seed: int = field(default_factory=lambda: int(_env("GEMINI_SEED", "7")))

    questions_path: Path = field(default_factory=lambda: _path("QUESTIONS_PATH", "data/questions.xlsx"))
    company_data_path: Path = field(default_factory=lambda: _path("COMPANY_DATA_PATH", "data/company_data.xlsx"))
    n_questions: int = field(default_factory=lambda: int(_env("N_QUESTIONS", "30")))

    top_k: int = field(default_factory=lambda: int(_env("RETRIEVAL_TOP_K", "25")))
    retrieval_strategy: str = field(default_factory=lambda: _env("RETRIEVAL_STRATEGY", "hybrid"))

    cache_dir: Path = field(default_factory=lambda: _path("GQE_CACHE_DIR", ".cache"))
    max_retries: int = field(default_factory=lambda: int(_env("GQE_MAX_RETRIES", "5")))
    runs_dir: Path = field(default_factory=lambda: _path("GQE_RUNS_DIR", "runs"))

    price_input_per_1m: float = field(default_factory=lambda: float(_env("PRICE_INPUT_PER_1M", "0")))
    price_output_per_1m: float = field(default_factory=lambda: float(_env("PRICE_OUTPUT_PER_1M", "0")))

    @property
    def mock(self) -> bool:
        """True when no live API calls should be made."""
        return _env("GQE_MOCK", "0") == "1" or not self.api_key


settings = Settings()
