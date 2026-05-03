"""
Phase 5 — Few-Shot RAG Store (Qdrant + nomic-embed-text)

WHY FEW-SHOT RAG:
  The Text-to-SQL and Text-to-Cypher LLM calls fail on complex questions because
  the model has no examples of what GOOD query generation looks like for THIS schema.

  Few-shot retrieval fixes this:
    1. We store verified (question → SQL/Cypher/answer) pairs in Qdrant
    2. At query time, we embed the new question and find the 3 most similar verified pairs
    3. We inject those pairs as examples into the LLM prompt
    4. The LLM generates much better SQL/Cypher because it has seen similar verified patterns

HOW IT WORKS:
  Embed: nomic-embed-text (local Ollama, 768-dim vectors, ~0.1s per question)
  Store: Qdrant (local file-based, no extra service needed)
  Retrieve: cosine similarity top-3

INTEGRATION POINTS:
  - text_to_sql.py: inject few-shot SQL examples into SQL generation prompt
  - text_to_cypher.py: inject few-shot Cypher examples into Cypher generation prompt
  - pipeline.py: retrieves examples before calling text_to_sql / text_to_cypher

EXPECTED IMPACT:
  Text-to-SQL accuracy:    ~55% → ~75%  (+20 points)
  Text-to-Cypher accuracy: ~40% → ~65%  (+25 points)
  Overall RAGAS:           ~55% → ~70%  (+15 points after Phase 4 fixes)
"""
