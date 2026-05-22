"""Swappable LLM adapter for multi-model baseline runs."""
from __future__ import annotations

import re
from typing import Protocol

import requests

from georgia_ev_intelligence.shared import config


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


def _clean_answer(answer: str) -> str:
    if not answer:
        return ""
    cleaned = answer.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:text|markdown)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned
