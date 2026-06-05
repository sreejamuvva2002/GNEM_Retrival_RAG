"""Closed-loop self-healing layer for the hybrid RAG pipeline.

retrieve -> judge -> generate -> verify -> bounded retry, with a best-effort
fail-safe. See georgia_ev_intelligence/runtime_pipeline/self_healing/loop.py.
"""
