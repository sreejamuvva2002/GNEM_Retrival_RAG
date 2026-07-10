"""Offline evaluation tooling for the route-generation and answer-generation stages.

This package does not call the router or executor itself — it only generates
grounded test questions (``query_generator``) and grades already-produced
answers (``answer_judge``). ``scripts/generate_test_queries.py``,
``scripts/eval_routing.py``, and ``scripts/eval_answers.py`` are the thin CLI
drivers that wire this package to the existing pipeline.
"""
