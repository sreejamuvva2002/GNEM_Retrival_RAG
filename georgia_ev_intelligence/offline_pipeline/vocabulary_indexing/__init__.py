"""Vocabulary term extraction and indexing for the KB offline pipeline."""

from .service import VocabularyIndexStats, index_vocabulary

__all__ = ["index_vocabulary", "VocabularyIndexStats"]
