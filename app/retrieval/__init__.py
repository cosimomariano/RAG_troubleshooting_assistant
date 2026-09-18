"""Interfacce e implementazioni per il recupero della conoscenza."""

from app.retrieval.base import Retriever
from app.retrieval.dense import DenseRetriever

__all__ = ["DenseRetriever", "Retriever"]
