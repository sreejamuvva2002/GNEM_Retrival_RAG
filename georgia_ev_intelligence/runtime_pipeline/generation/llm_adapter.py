"""Swappable LLM adapter for multi-model baseline runs.

WHY THIS FILE EXISTS
--------------------
When ``run_baseline.py`` loops over multiple Ollama models, each iteration
needs to send requests to a DIFFERENT model name.  ``OllamaAdapter`` wraps a
single model name and exposes a ``generate(prompt, timeout) -> str`` method,
allowing the same pipeline classes (OnlyRagAnswerPipeline, etc.) to work
unchanged across all models by injecting the adapter at construction time.

DESIGN
------
``LLMAdapter`` is a ``typing.Protocol`` defining the interface: any object with
a ``model_name: str`` attribute and a ``generate(prompt, timeout)`` method
qualifies.  ``OllamaAdapter`` is the concrete implementation for local Ollama.

DIFFERENCES FROM ``llm_client.generate_answer``
------------------------------------------------
- ``llm_client.generate_answer`` uses ``config.OLLAMA_LLM_MODEL`` (fixed at
  module load time).
- ``OllamaAdapter`` accepts a ``model_name`` at instantiation, enabling
  per-loop model switching in ``run_baseline.py``.
- Both use identical Ollama API parameters (temperature, top_p, num_predict)
  and the same ``_clean_answer()`` cleaning step.

CORRECTNESS CONTRACT
--------------------
- ``OllamaAdapter.generate()`` re-uses the connection per call (no keep-alive
  session); Ollama handles concurrent requests on localhost.
- The ``_clean_answer`` import from ``llm_client`` ensures consistent artifact
  removal across both the default client and the adapter path.
"""
from __future__ import annotations

from typing import Protocol

import requests

from georgia_ev_intelligence.shared import config
from georgia_ev_intelligence.runtime_pipeline.generation.llm_client import _clean_answer


class LLMAdapter(Protocol):
    """Generate text from a prompt."""

    model_name: str

    def generate(self, prompt: str, timeout: int = 180) -> str:
        """Return the generated text."""


class OllamaAdapter:
    """Call a locally-running Ollama model by name."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        num_predict: int | None = None,
    ) -> None:
        self.model_name = model_name
        self._base_url = base_url or config.OLLAMA_BASE_URL
        self._temperature = temperature if temperature is not None else config.OLLAMA_TEMPERATURE
        self._top_p = top_p if top_p is not None else config.OLLAMA_TOP_P
        self._num_predict = num_predict if num_predict is not None else config.OLLAMA_NUM_PREDICT

    def generate(self, prompt: str, timeout: int = 180) -> str:
        resp = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self._temperature,
                    "top_p": self._top_p,
                    "num_predict": self._num_predict,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        return _clean_answer(answer)
