"""
Shared configuration loader.
Reads .env and config/settings.yaml. Used by every phase.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import yaml
from dotenv import load_dotenv


def _find_env_file() -> Path:
    """Walk up from this file until we find .env"""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = current / ".env"
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError(
        "Could not find .env file. Make sure it exists in georgia_ev_intelligence/"
    )


def _find_settings_file() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = current / "config" / "settings.yaml"
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError("Could not find config/settings.yaml")


def _dict_to_namespace(data: dict) -> SimpleNamespace:
    """Recursively convert nested dicts to SimpleNamespace for dot access."""
    ns = SimpleNamespace()
    for key, value in data.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def load_settings() -> SimpleNamespace:
    """Load settings.yaml as a SimpleNamespace for dot-access."""
    settings_path = _find_settings_file()
    with settings_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _dict_to_namespace(raw or {})


class Config:
    """Central config object. Access via Config.get()."""
    _instance: "Config | None" = None

    def __init__(self) -> None:
        env_path = _find_env_file()
        load_dotenv(env_path, override=False)
        self._settings = load_settings()

    # ── Tavily ───────────────────────────────────────────────
    @property
    def tavily_api_key(self) -> str:
        return self._require("TAVILY_API_KEY")

    # ── Qdrant ───────────────────────────────────────────────
    @property
    def qdrant_url(self) -> str:
        return self._require("QDRANT_URL")

    @property
    def qdrant_api_key(self) -> str:
        return self._require("QDRANT_API_KEY")

    @property
    def qdrant_collection_base(self) -> str:
        return os.environ.get("QDRANT_COLLECTION_NAME", "georgia_ev_chunks")

    @staticmethod
    def _sanitize_collection_part(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "default"

    @property
    def qdrant_collection(self) -> str:
        embed_model = self._sanitize_collection_part(self.ollama_embed_model)
        return f"{self.qdrant_collection_base}__{embed_model}"

    @property
    def qdrant_dense_name(self) -> str:
        return os.environ.get("QDRANT_DENSE_VECTOR_NAME", "dense")

    @property
    def qdrant_sparse_name(self) -> str:
        return os.environ.get("QDRANT_SPARSE_VECTOR_NAME", "sparse")

    @property
    def qdrant_dimensions(self) -> int:
        return int(os.environ.get("QDRANT_VECTOR_DIMENSIONS", "768"))

    # ── Neo4j ────────────────────────────────────────────────
    @property
    def neo4j_uri(self) -> str:
        return self._require("NEO4J_URI")

    @property
    def neo4j_username(self) -> str:
        return os.environ.get("NEO4J_USERNAME", "neo4j")

    @property
    def neo4j_password(self) -> str:
        return self._require("NEO4J_PASSWORD")

    # ── PostgreSQL ───────────────────────────────────────────
    @property
    def database_url(self) -> str:
        return self._require("DATABASE_URL")

    # ── Backblaze B2 ─────────────────────────────────────────
    @property
    def b2_bucket(self) -> str:
        return self._require("B2_BUCKET_NAME")

    @property
    def b2_endpoint(self) -> str:
        return self._require("B2_ENDPOINT_URL")

    @property
    def b2_region(self) -> str:
        return os.environ.get("B2_REGION", "us-east-005")

    @property
    def b2_access_key(self) -> str:
        return self._require("B2_ACCESS_KEY_ID")

    @property
    def b2_secret_key(self) -> str:
        return self._require("B2_SECRET_ACCESS_KEY")

    # ── Ollama ───────────────────────────────────────────────
    @property
    def ollama_base_url(self) -> str:
        return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def ollama_llm_model(self) -> str:
        return os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:14b")

    @property
    def ollama_cypher_model(self) -> str:
        """Separate model for Cypher generation — faster/code-focused (Gemma)."""
        return os.environ.get("OLLAMA_CYPHER_MODEL", "gemma3:4b")

    @property
    def ollama_llm_fallback(self) -> str:
        return os.environ.get("OLLAMA_LLM_FALLBACK", "llama3.1:8b")

    @property
    def ollama_embed_model(self) -> str:
        return os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    @property
    def reranker_model(self) -> str:
        return os.environ.get(
            "CROSS_ENCODER_RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L12-v2",
        )

    @property
    def reranker_max_candidates(self) -> int:
        return int(os.environ.get("CROSS_ENCODER_RERANK_TOP_K", "48"))


    # ── Settings YAML ────────────────────────────────────────
    @property
    def settings(self) -> SimpleNamespace:
        return self._settings

    # ── Internal ─────────────────────────────────────────────
    @staticmethod
    def _require(key: str) -> str:
        value = os.environ.get(key, "")
        if not value or "XXXXX" in value:
            raise EnvironmentError(
                f"Required environment variable '{key}' is not set. "
                f"Please update georgia_ev_intelligence/.env"
            )
        return value

    @classmethod
    def get(cls) -> "Config":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Force reload — useful in tests."""
        cls._instance = None
