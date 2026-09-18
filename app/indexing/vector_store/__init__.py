"""Contratti e implementazioni per la persistenza vettoriale."""

from app.indexing.vector_store.base import PersistentVectorIndex
from app.indexing.vector_store.faiss_store import FaissVectorIndex

__all__ = ["FaissVectorIndex", "PersistentVectorIndex"]
