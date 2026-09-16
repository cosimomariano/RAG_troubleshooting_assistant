"""Strategie di suddivisione dei documenti in chunk recuperabili."""

from app.ingestion.chunking.section_aware import SectionAwareChunker

__all__ = ["SectionAwareChunker"]
