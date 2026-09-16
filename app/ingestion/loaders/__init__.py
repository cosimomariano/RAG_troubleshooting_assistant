"""Loader supportati dalla pipeline di ingestion."""

from app.ingestion.loaders.local import SUPPORTED_EXTENSIONS, LocalDocumentLoader

__all__ = ["SUPPORTED_EXTENSIONS", "LocalDocumentLoader"]
