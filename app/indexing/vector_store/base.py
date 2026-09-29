"""Contratto per la persistenza dell'indice vettoriale."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

from app.indexing.embeddings.base import EmbeddingVector
from app.models.documents import DocumentChunk

VectorSearchMatch = tuple[int, float]


@runtime_checkable
class PersistentVectorIndex(Protocol):
    """Definisce le operazioni minime di un indice vettoriale persistente."""

    def getDimension(self) -> int: ...

    def getSize(self) -> int: ...

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[EmbeddingVector],
    ) -> None: ...

    def getChunk(self, position: int) -> DocumentChunk: ...

    def search(self, vector: EmbeddingVector, k: int) -> list[VectorSearchMatch]: ...

    def save(self, directoryPath: Path) -> None: ...

    @classmethod
    def load(cls, directoryPath: Path) -> Self: ...
