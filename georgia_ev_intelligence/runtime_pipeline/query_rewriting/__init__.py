"""Query rewriting subsystem for structured vocabulary-based retrieval.

Parses raw user questions into structured filters via LLM, then matches
those filters against the kb_vocabulary_terms table to resolve parent row_ids.
"""
