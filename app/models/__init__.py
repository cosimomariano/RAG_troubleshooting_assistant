"""Modelli di dominio condivisi dalla pipeline RAG."""

from app.models.documents import Document, DocumentChunk, SourceMetadata
from app.models.rag import RAGResponse, SourceReference
from app.models.retrieval import RetrievalResult

__all__ = [
    "Document",
    "DocumentChunk",
    "RAGResponse",
    "RetrievalResult",
    "SourceMetadata",
    "SourceReference",
]