"""Modelli di dominio condivisi dalla pipeline RAG."""

from app.models.base import StrictModel
from app.models.documents import Document, DocumentChunk, SourceMetadata
from app.models.rag import RAGResponse, SourceReference
from app.models.retrieval import RetrievalContribution, RetrievalResult

__all__ = [
    "Document",
    "DocumentChunk",
    "RAGResponse",
    "RetrievalContribution",
    "RetrievalResult",
    "SourceMetadata",
    "SourceReference",
    "StrictModel",
]
