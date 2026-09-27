"""Assemblaggio dei componenti eseguibili del sistema."""

from app.bootstrap.application import RAGApplicationFactory
from app.bootstrap.knowledge_base import (
    IndexingReport,
    KnowledgeBaseIndexer,
    KnowledgeBaseProcessor,
)

__all__ = [
    "IndexingReport",
    "KnowledgeBaseIndexer",
    "KnowledgeBaseProcessor",
    "RAGApplicationFactory",
]
