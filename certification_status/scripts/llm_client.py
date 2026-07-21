"""Minimal client for a local LLM served through an OpenAI-compatible endpoint.

Works out of the box with Ollama (http://localhost:11434/v1) and also with
LM Studio, llama.cpp server, or vLLM — anything speaking /v1/chat/completions.
Configure via environment variables (or a .env file loaded by the caller):

    LLM_BASE_URL   default http://localhost:11434/v1
    LLM_MODEL      default qwen3:14b
    LLM_API_KEY    default "ollama" (Ollama ignores it; other servers may not)
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen3:14b")
API_KEY = os.environ.get("LLM_API_KEY", "ollama")


def chat(messages: list[dict], model: str | None = None, base_url: str | None = None,
         temperature: float = 0.0, timeout: float = 300.0) -> str:
    """Send a chat completion request and return the assistant message text."""
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response.

    Local models sometimes wrap JSON in markdown fences or <think> blocks
    (Qwen3); strip those before parsing.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object in model response: {text[:200]!r}")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"Unbalanced JSON in model response: {text[:200]!r}")


def ping(base_url: str | None = None) -> bool:
    """Return True if the LLM server is reachable."""
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False
