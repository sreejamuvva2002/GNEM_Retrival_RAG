"""Runtime LLM answer generation.

This package provides:
  - ``llm_client``  — ``generate_answer(prompt, timeout)`` using config model
  - ``llm_adapter`` — ``OllamaAdapter`` for per-model multi-run inference

The adapter is used by ``run_baseline.py`` to swap models per loop iteration.
The client is the default injected into pipeline classes when no adapter is given.
"""
