"""Componenti per il caricamento delle fonti documentali."""

from app.ingestion.chunking import SectionAwareChunker
from app.ingestion.loaders import LocalDocumentLoader

__all__ = ["LocalDocumentLoader", "SectionAwareChunker"]
