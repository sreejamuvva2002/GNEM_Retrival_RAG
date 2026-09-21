"""Thin, cacheable wrapper around the Gemini API.

Three things this adds over calling ``google-genai`` directly:

* **Disk cache** - identical (model, prompt, config) requests are answered from
  ``.cache/`` so a re-run of the evaluation costs nothing.
* **Retries** - exponential backoff on 429/5xx, which you will hit when
  scoring 30 questions x 3 modes x several judges.
* **Mock mode** - with no ``GEMINI_API_KEY`` (or ``GQE_MOCK=1``) every call
  returns a deterministic stub, so the full pipeline and the tests run offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import Settings, settings as default_settings

log = logging.getLogger(__name__)

_TRANSIENT = ("429", "500", "502", "503", "504", "deadline", "unavailable", "overloaded")


@dataclass
class GenResult:
    """One generation, plus everything the evaluation needs to score it."""

    text: str
    model: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    web_sources: list[dict[str, str]] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    cached: bool = False
    mocked: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "web_sources": self.web_sources,
            "search_queries": self.search_queries,
            "cached": self.cached,
            "mocked": self.mocked,
            "error": self.error,
        }


def _key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class GeminiClient:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or default_settings
        self.cache_dir = Path(self.cfg.cache_dir)
        (self.cache_dir / "gen").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "embed").mkdir(parents=True, exist_ok=True)
        self._client = None

    # -- plumbing ---------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            from google import genai  # imported lazily so mock mode needs no SDK

            self._client = genai.Client(api_key=self.cfg.api_key)
        return self._client

    def _cached(self, kind: str, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / kind / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _store(self, kind: str, key: str, value: dict[str, Any]) -> None:
        try:
            (self.cache_dir / kind / f"{key}.json").write_text(json.dumps(value))
        except OSError as exc:  # a broken cache must never fail a run
            log.warning("cache write failed: %s", exc)

    # -- generation -------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        use_web_search: bool = False,
        response_schema: Any = None,
        max_output_tokens: int | None = None,
        use_cache: bool = True,
    ) -> GenResult:
        model = model or self.cfg.model
        temperature = self.cfg.temperature if temperature is None else temperature
        max_output_tokens = max_output_tokens or self.cfg.max_output_tokens

        cache_key = _key(
            {
                "prompt": prompt,
                "system": system,
                "model": model,
                "temperature": temperature,
                "web": use_web_search,
                "schema": str(response_schema),
                "max_tokens": max_output_tokens,
                "seed": self.cfg.seed,
            }
        )
        if use_cache:
            hit = self._cached("gen", cache_key)
            if hit is not None:
                return GenResult(**{**hit, "cached": True})

        if self.cfg.mock:
            result = _mock_generate(prompt, model, response_schema)
        else:
            result = self._generate_live(
                prompt,
                system=system,
                model=model,
                temperature=temperature,
                use_web_search=use_web_search,
                response_schema=response_schema,
                max_output_tokens=max_output_tokens,
            )

        if use_cache and result.error is None:
            self._store("gen", cache_key, result.to_dict())
        return result

    def _generate_live(
        self,
        prompt: str,
        *,
        system: str | None,
        model: str,
        temperature: float,
        use_web_search: bool,
        response_schema: Any,
        max_output_tokens: int,
    ) -> GenResult:
        from google.genai import types

        cfg_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if system:
            cfg_kwargs["system_instruction"] = system
        if use_web_search:
            # Grounding with Google Search: the model decides when to search,
            # and the response carries the sources it used.
            cfg_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        elif response_schema is not None:
            # Structured output and tools cannot be combined on one call.
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_schema"] = response_schema

        last_error = "unknown error"
        for attempt in range(self.cfg.max_retries):
            started = time.perf_counter()
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
                latency = (time.perf_counter() - started) * 1000
                usage = getattr(response, "usage_metadata", None)
                sources, queries = _grounding(response)
                return GenResult(
                    text=(response.text or "").strip(),
                    model=model,
                    latency_ms=latency,
                    input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                    output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                    web_sources=sources,
                    search_queries=queries,
                )
            except Exception as exc:  # noqa: BLE001 - SDK raises many error types
                last_error = f"{type(exc).__name__}: {exc}"
                if not any(token in str(exc).lower() for token in _TRANSIENT):
                    break
                sleep_for = min(2**attempt, 32) + random.random()
                log.warning("gemini call failed (%s), retrying in %.1fs", last_error, sleep_for)
                time.sleep(sleep_for)

        return GenResult(text="", model=model, error=last_error)

    def generate_json(self, prompt: str, *, schema: Any, model: str | None = None, **kwargs) -> dict[str, Any]:
        """Generate and parse a JSON object, returning ``{}`` on failure."""
        result = self.generate(prompt, response_schema=schema, model=model, **kwargs)
        if result.error or not result.text:
            return {}
        try:
            parsed = json.loads(result.text)
        except json.JSONDecodeError:
            start, end = result.text.find("{"), result.text.rfind("}")
            if start == -1 or end <= start:
                return {}
            try:
                parsed = json.loads(result.text[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return parsed if isinstance(parsed, dict) else {}

    # -- embeddings -------------------------------------------------------
    def embed(self, texts: Sequence[str], *, task_type: str = "SEMANTIC_SIMILARITY"):
        import numpy as np

        vectors: list[list[float]] = []
        pending: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            key = _key({"text": text, "model": self.cfg.embed_model, "task": task_type})
            hit = self._cached("embed", key)
            if hit is None:
                vectors.append([])
                pending.append((index, text))
            else:
                vectors.append(hit["v"])

        for index, text in pending:
            key = _key({"text": text, "model": self.cfg.embed_model, "task": task_type})
            vector = _mock_embed(text) if self.cfg.mock else self._embed_live(text, task_type)
            vectors[index] = vector
            self._store("embed", key, {"v": vector})

        matrix = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-9, None)

    def _embed_live(self, text: str, task_type: str) -> list[float]:
        from google.genai import types

        for attempt in range(self.cfg.max_retries):
            try:
                response = self.client.models.embed_content(
                    model=self.cfg.embed_model,
                    contents=text,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                return list(response.embeddings[0].values)
            except Exception as exc:  # noqa: BLE001
                if not any(token in str(exc).lower() for token in _TRANSIENT):
                    log.error("embedding failed: %s", exc)
                    break
                time.sleep(min(2**attempt, 32) + random.random())
        return _mock_embed(text)


def _grounding(response) -> tuple[list[dict[str, str]], list[str]]:
    """Pull web sources and issued search queries out of a grounded response."""
    sources: list[dict[str, str]] = []
    queries: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        metadata = getattr(candidate, "grounding_metadata", None)
        if metadata is None:
            continue
        queries.extend(getattr(metadata, "web_search_queries", None) or [])
        for chunk in getattr(metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web is not None:
                sources.append({"title": getattr(web, "title", "") or "", "uri": getattr(web, "uri", "") or ""})
    seen, unique = set(), []
    for source in sources:
        if source["uri"] and source["uri"] not in seen:
            seen.add(source["uri"])
            unique.append(source)
    return unique, list(dict.fromkeys(queries))


def _mock_generate(prompt: str, model: str, response_schema: Any = None) -> GenResult:
    """Deterministic stand-in so the pipeline is exercisable without a key.

    For free-text calls it echoes a little of the supplied context, which keeps
    the context-grounded metrics non-degenerate while staying obviously fake.
    For structured calls it fabricates a schema-shaped object, so the judge and
    RAG-metric code paths are exercised end to end offline.
    """
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if response_schema is not None:
        text = json.dumps(_mock_json(response_schema, digest))
        return GenResult(text=text, model=model, latency_ms=1.0, mocked=True,
                         input_tokens=len(prompt) // 4, output_tokens=len(text) // 4)
    echoed = [line.strip() for line in prompt.splitlines() if line.strip().startswith("[C")][:3]
    body = " ".join(echoed) if echoed else "No internal context was supplied for this question."
    text = f"[MOCK {digest[:8]}] {body}"
    return GenResult(
        text=text,
        model=model,
        latency_ms=1.0,
        input_tokens=len(prompt) // 4,
        output_tokens=len(text) // 4,
        mocked=True,
    )


def _mock_json(schema: Any, digest: str) -> Any:
    """Build a deterministic value matching a (JSON-schema-shaped) dict."""
    rng = random.Random(digest)
    if not isinstance(schema, dict):
        return {}
    kind = schema.get("type")
    if kind == "object":
        return {name: _mock_json(sub, digest + name) for name, sub in (schema.get("properties") or {}).items()}
    if kind == "array":
        item = schema.get("items", {"type": "string"})
        return [_mock_json(item, digest + str(i)) for i in range(2)]
    if kind == "integer":
        return rng.randint(2, 5)
    if kind == "number":
        return round(rng.uniform(0, 1), 3)
    if kind == "boolean":
        return rng.random() > 0.35
    choices = schema.get("enum")
    return rng.choice(choices) if choices else f"mock-{digest[:6]}"


def _mock_embed(text: str, dim: int = 256) -> list[float]:
    rng = random.Random(hashlib.sha256(text.encode()).hexdigest())
    return [rng.gauss(0, 1) for _ in range(dim)]
