"""Interfacce e adapter per la generazione degli embedding."""

from app.indexing.embeddings.base import EmbeddingModel, EmbeddingVector
from app.indexing.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel

__all__ = ["EmbeddingModel", "EmbeddingVector", "SentenceTransformerEmbeddingModel"]
