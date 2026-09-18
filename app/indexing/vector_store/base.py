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

    @property
    def dimension(self) -> int: ...

    @property
    def size(self) -> int: ...

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[EmbeddingVector],
    ) -> None: ...

    def get_chunk(self, position: int) -> DocumentChunk: ...

    def search(self, vector: EmbeddingVector, k: int) -> list[VectorSearchMatch]: ...

    def save(self, directory_path: Path) -> None: ...

    @classmethod
    def load(cls, directory_path: Path) -> Self: ...
