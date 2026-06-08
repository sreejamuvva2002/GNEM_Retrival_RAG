"""Single config surface for the route_generation subpackage.

Re-exports the relevant values from ``georgia_ev_intelligence.shared.config`` so
routing modules import from one place (``from ..config import ...``).
``shared.config.settings`` remains the single source of truth; nothing new is
read from the environment here.
"""
from __future__ import annotations

from georgia_ev_intelligence.shared import config as _shared

# LLM — reused from the shared Ollama settings (README OLLAMA_MODEL == OLLAMA_LLM_MODEL).
OLLAMA_BASE_URL: str = _shared.OLLAMA_BASE_URL
OLLAMA_LLM_MODEL: str = _shared.OLLAMA_LLM_MODEL
OLLAMA_TEMPERATURE: float = _shared.OLLAMA_TEMPERATURE

# Router behaviour.
ROUTER_CONFIDENCE_THRESHOLD: float = _shared.ROUTER_CONFIDENCE_THRESHOLD
PRE_ROUTER_HIGH_CONFIDENCE: float = _shared.PRE_ROUTER_HIGH_CONFIDENCE
ROUTER_TEMPERATURE: float = _shared.ROUTER_TEMPERATURE
ROUTER_NUM_PREDICT: int = _shared.ROUTER_NUM_PREDICT

# Metadata provider selection.
METADATA_PROVIDER: str = _shared.METADATA_PROVIDER          # "live" | "file"
METADATA_SNAPSHOT_PATH: str = _shared.METADATA_SNAPSHOT_PATH

__all__ = [
    "OLLAMA_BASE_URL",
    "OLLAMA_LLM_MODEL",
    "OLLAMA_TEMPERATURE",
    "ROUTER_CONFIDENCE_THRESHOLD",
    "PRE_ROUTER_HIGH_CONFIDENCE",
    "ROUTER_TEMPERATURE",
    "ROUTER_NUM_PREDICT",
    "METADATA_PROVIDER",
    "METADATA_SNAPSHOT_PATH",
]
