import os

# Every test runs offline and deterministically.
os.environ.setdefault("GQE_MOCK", "1")
os.environ.setdefault("RETRIEVAL_STRATEGY", "bm25")
