"""LLM client for data augmentation - supports local and API-based models."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Optional

import requests

from . import config


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text from a prompt."""
        pass


class OllamaClient(LLMClient):
    """Client for local Ollama-hosted models."""

    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        model: str = config.DATA_GEN_OLLAMA_MODEL,
        temperature: float = config.DATA_GEN_TEMPERATURE,
        timeout: int = config.DATA_GEN_TIMEOUT,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using Ollama."""
        if max_tokens is None:
            max_tokens = config.DATA_GEN_MAX_TOKENS

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")


class VLLMClient(LLMClient):
    """Client for vLLM-hosted models."""

    def __init__(
        self,
        base_url: str = config.VLLM_BASE_URL,
        model: str = config.DATA_GEN_VLLM_MODEL,
        temperature: float = config.DATA_GEN_TEMPERATURE,
        timeout: int = config.DATA_GEN_TIMEOUT,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using vLLM OpenAI-compatible API."""
        if max_tokens is None:
            max_tokens = config.DATA_GEN_MAX_TOKENS

        try:
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_tokens": max_tokens,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["text"].strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"vLLM request failed: {e}")


class OpenAIClient(LLMClient):
    """Client for OpenAI API."""

    def __init__(
        self,
        api_key: str = config.OPENAI_API_KEY,
        model: str = config.DATA_GEN_OPENAI_MODEL,
        temperature: float = config.DATA_GEN_TEMPERATURE,
        timeout: int = config.DATA_GEN_TIMEOUT,
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using OpenAI API."""
        if max_tokens is None:
            max_tokens = config.DATA_GEN_MAX_TOKENS

        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            raise RuntimeError("openai package not installed")
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}")


class GeminiClient(LLMClient):
    """Client for Google Gemini API."""

    def __init__(
        self,
        api_key: str = config.GOOGLE_API_KEY,
        model: str = config.DATA_GEN_GEMINI_MODEL,
        temperature: float = config.DATA_GEN_TEMPERATURE,
    ):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using Google Gemini API."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=max_tokens or config.DATA_GEN_MAX_TOKENS,
            )
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt, generation_config=generation_config)
            return response.text.strip()
        except ImportError:
            raise RuntimeError("google-generativeai package not installed")
        except Exception as e:
            raise RuntimeError(f"Gemini request failed: {e}")


def get_client(provider: Optional[str] = None) -> LLMClient:
    """Factory function to get the appropriate LLM client."""
    if provider is None:
        provider = config.DATA_GEN_LLM_PROVIDER

    provider = provider.lower()

    if provider == "ollama":
        return OllamaClient()
    elif provider == "vllm":
        return VLLMClient()
    elif provider == "openai":
        return OpenAIClient()
    elif provider == "gemini":
        return GeminiClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
